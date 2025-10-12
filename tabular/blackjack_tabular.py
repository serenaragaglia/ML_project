import gymnasium as gym
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

#hyperparameters
EPISODES_NUM = 50000
ALPHA = 0.1
GAMMA = 0.9
MIN_EPS = 0.01
EPS_DECAY = 0.999

#epsilon greedy policy
def epsilon_greedy(state, epsilon, q_table, env):
    n = random.random()
    if n < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

#tabular q-learning algorithm
def tabular_qlearning(env, policy_file, episodes_num = EPISODES_NUM, alpha = ALPHA, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0):

    q_table = defaultdict(lambda : np.zeros(env.action_space.n))

    rewards_per_episode = []
    epsilon_per_episode = []

    for episode in range(episodes_num):
        state, _ = env.reset()
        finished = False 
        total_reward = 0

        while not finished:
            action = epsilon_greedy(state, epsilon, q_table, env)

            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated            

            best_action = np.argmax(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_action] - q_table[state][action])

            total_reward += reward
            state = next_state
        
        rewards_per_episode.append(total_reward)
        epsilon_per_episode.append(epsilon)
        epsilon = max(MIN_EPS, epsilon * eps_decay)

        if(episode + 1 ) % 100 == 0 : 
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1} - Average Reward: {avg_reward:.2f}")

    with open(policy_file, "wb") as f:
        pickle.dump(dict(q_table), f)

    return rewards_per_episode, epsilon_per_episode

#run optimal policy
def run_policy(env, policy_file):
    with open(policy_file, "rb") as f:
        q =  defaultdict(lambda: np.zeros(env.action_space.n),pickle.load(f))
    
    rewards = []
    for _ in range (EPISODES_NUM):
        state, _ = env.reset()
        finished = False
        tot_reward = 0

        while not finished:
            action = np.argmax(q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            tot_reward += reward
        
        rewards.append(tot_reward)
    print(f"Average Reward for optimal policy: {np.mean(rewards):.2f}")

    return rewards

#pick random actions
def random_episodes(env):
    rewards = []
    for _ in range (EPISODES_NUM):
        state, _ = env.reset()
        finished = False
        tot_reward = 0

        while not finished:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            tot_reward += reward

        rewards.append(tot_reward)

    print(f"Average Reward for Random action: {np.mean(rewards):.2f}")
    return rewards

def plot_random_vs_greedy(random_rewards, greedy_rewards, epsilon):
    window = 100
    avg_greedy = np.convolve(greedy_rewards, np.ones(window)/window, mode='valid')
    avg_random = np.convolve(random_rewards, np.ones(window)/window, mode='valid') 

    fig, ax1 = plt.subplots(figsize = (10,5))
    ax1.plot(avg_greedy, label = "Greedy", color ='blue')
    ax1.plot(avg_random, label = "Random", color = 'brown')

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average reward")
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(epsilon, label = "Epsilon", color = 'green', linestyle='--')
    ax2.set_ylabel("Epsilon")
    ax2.legend(loc='upper right')

    plt.title("Learning trend")

    plt.show()

def plot_per_policy(rewards, policy):
    episodes = range(len(rewards))
    mean_rewards = np.mean(rewards)

    plt.scatter(episodes, rewards)
    plt.title(f"Reward per episode - {policy}")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.axhline(y = mean_rewards, color ='red', label = f'Rewards Mean = {mean_rewards:.2f}')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

def plot_hypeparameters(res1, res2, res3):
    window = 100
    plt.figure(figsize=(10,5))

    avg1 = np.convolve(res1, np.ones(window)/window, mode='valid' )
    avg2 = np.convolve(res2, np.ones(window)/window, mode='valid' )
    avg3 = np.convolve(res3, np.ones(window)/window, mode='valid' )

    plt.plot(avg1, color = 'blue', label = "First")
    plt.plot(avg2, color = 'orange', label = "Second")
    plt.plot(avg3, color = 'pink', label = "Third")

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Trend for different hyperameters')

    plt.grid(axis='y')

    plt.legend(loc='upper left')
    
    
    plt.show()
    
if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=False, sab=False)

    mode = input("Choose the modality : 0 = training, 1 = running policy, 2 = tuning with different hyperparameters  ").strip()
    policy_file = input("File policy name (empty fot default): ").strip()
    if policy_file == "":
        policy_file = "default.pkl"

    if mode == "0":
        rewards, epsilon = tabular_qlearning(env, policy_file, episodes_num = EPISODES_NUM, alpha = ALPHA, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0)
        ran_rewards = random_episodes(env)
        plot_random_vs_greedy(ran_rewards, rewards, epsilon)
    elif mode == "1":
        ran_rewards = random_episodes(env)
        greedy_rewards = run_policy(env, policy_file)
        plot_per_policy(ran_rewards, 'Random')
        plot_per_policy(greedy_rewards, 'Greedy')
    elif mode == "2":
        r1, eps1 = tabular_qlearning(env, policy_file = ("tuning_first.pkl"), episodes_num = 10000, alpha=0.1, gamma= 0.99, eps_decay=0.995, epsilon=1.0)
        r2, eps2 = tabular_qlearning(env, policy_file = ("tuning_second.pkl"), episodes_num = 10000, alpha=0.1, gamma= 0.9, eps_decay=0.99, epsilon=1.0)
        r3, eps3 = tabular_qlearning(env, policy_file = ("tuning_third.pkl"), episodes_num = 10000, alpha=0.1, gamma= 0.9, eps_decay=0.9, epsilon=1.0)
        plot_hypeparameters(r1, r2, r3)
    else: print("Error")

