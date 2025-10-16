import gymnasium as gym
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

#hyperparameters
EPISODES_NUM = 10000
ALPHA = 0.1
GAMMA = 0.999
MIN_EPS = 0.1
EPS_DECAY = 0.9999

#https://gymnasium.farama.org/v0.26.3/tutorials/blackjack_tutorial/

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

    wins = []
    losses = []

    win_rate_tot = []
    loss_rate_tot = []

    for episode in range(episodes_num):
        state, _ = env.reset()
        finished = False 
        total_reward = 0

        while not finished:
            action = epsilon_greedy(state, epsilon, q_table, env)

            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated            

            best_action = np.argmax(q_table[next_state])

            if finished:
                target = reward
            else: 
                target = reward + gamma * q_table[next_state][best_action]

            q_table[state][action] += alpha * (target - q_table[state][action])

            total_reward += reward
            state = next_state
        
        rewards_per_episode.append(total_reward)

        if total_reward == 1:
            wins.append(1)
        elif total_reward == -1:
            losses.append(-1)

        epsilon_per_episode.append(epsilon)
        epsilon = max(MIN_EPS, epsilon * eps_decay)

        if(episode + 1 ) % 100 == 0 : 
            avg_reward = np.mean(rewards_per_episode[-100:])
            win_rate = np.mean([r == 1 for r in rewards_per_episode]) * 100
            lose_rate = np.mean([r == -1 for r in rewards_per_episode]) * 100

            win_rate_tot.append(win_rate)
            loss_rate_tot.append(lose_rate)

            print(f"Ep {episode+1:6d} | AvgReward: {avg_reward:+.3f} | "
              f"WinRate: {win_rate:.1f}% | LoseRate: {lose_rate:.1f}% | Eps: {epsilon:.3f}")

    with open(policy_file, "wb") as f:
        pickle.dump(dict(q_table), f)

    return rewards_per_episode, epsilon_per_episode, win_rate_tot, loss_rate_tot

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
    window = 1000
    avg_greedy = np.convolve(greedy_rewards, np.ones(window)/window, mode='valid')
    avg_random = np.convolve(random_rewards, np.ones(window)/window, mode='valid') 

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(avg_greedy, label="Greedy (Q-learning)", color='royalblue', linewidth=2)
    ax1.plot(avg_random, label="Random", color='darkorange', linestyle='--', linewidth=2)

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Smoothed Average Reward", fontsize=12)
    ax1.set_title("Q-learning vs Random Policy", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(epsilon[:len(avg_greedy)], label="Epsilon decay", color='green', linestyle=':', linewidth=2)
    ax2.set_ylabel("Epsilon", fontsize=12)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_training_stats(win_rate_history, lose_rate_history, avg_reward_history, stats_every):
    episodes = np.arange(stats_every, stats_every * len(win_rate_history) + 1, stats_every)

    plt.figure(figsize=(12,6))
    plt.plot(episodes, win_rate_history, label='Win Rate (%)', color='green', linewidth=2)
    plt.plot(episodes, lose_rate_history, label='Lose Rate (%)', color='red', linewidth=2)
    #plt.plot(episodes, avg_reward_history, label='Avg Reward', color='blue', linestyle='--')

    plt.title("Training Performance (Q-learning)", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Percentage / Reward", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_per_policy(rewards, policy):
    window = 500
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    std = np.std(rewards)
    episodes = np.arange(len(smoothed))

    plt.figure(figsize=(10,5))
    plt.plot(episodes, smoothed, label=f'{policy} (moving avg)', color='royalblue')
    plt.fill_between(episodes, smoothed-std, smoothed+std, color='blue', alpha=0.1)
    plt.axhline(y=np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.title(f'Reward Trend - {policy}', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_hypeparameters(res1, res2, res3):
    window = 200  # finestra più grande per media mobile
    step = 10     # campioniamo ogni 10 episodi per leggibilità
    
    # calcolo media mobile
    avg1 = np.convolve(res1, np.ones(window)/window, mode='valid')[::step]
    avg2 = np.convolve(res2, np.ones(window)/window, mode='valid')[::step]
    avg3 = np.convolve(res3, np.ones(window)/window, mode='valid')[::step]

    plt.figure(figsize=(12,6))
    plt.plot(avg1, color='blue', label="First", linewidth=2, alpha=0.8)
    plt.plot(avg2, color='orange', label="Second", linewidth=2, alpha=0.8)
    plt.plot(avg3, color='pink', label="Third", linewidth=2, alpha=0.8)

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Trend for different hyperparameters')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.show()
    
if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=True, sab=False)

    mode = input("Choose the modality : 0 = training, 1 = running policy, 2 = tuning with different hyperparameters  ").strip()
    policy_file = input("File policy name (empty fot default): ").strip()
    if policy_file == "":
        policy_file = "default.pkl"

    if mode == "0":
        rewards, epsilon, win, loss = tabular_qlearning(env, policy_file, episodes_num = EPISODES_NUM, alpha = ALPHA, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0)
        ran_rewards = random_episodes(env)
        avg_reward = np.mean(rewards)
        print(f"Greedy mean: {np.mean(rewards):.2f}")
        print(f"Random mean: {np.mean(ran_rewards):.2f}")
        plot_random_vs_greedy(ran_rewards, rewards, epsilon)
        plot_training_stats(win, loss, avg_reward, 100)
    elif mode == "1":
        ran_rewards = random_episodes(env)
        greedy_rewards = run_policy(env, policy_file)
        plot_per_policy(ran_rewards, 'Random')
        plot_per_policy(greedy_rewards, 'Greedy')
    elif mode == "2":
        r1, eps1, _, _ = tabular_qlearning(env, policy_file = ("tuning_first.pkl"), episodes_num = 10000, alpha=0.1, gamma= 0.99, eps_decay=0.995, epsilon=1.0)
        r2, eps2, _, _ = tabular_qlearning(env, policy_file = ("tuning_second.pkl"), episodes_num = 10000, alpha=0.1, gamma= 0.9, eps_decay=0.99, epsilon=1.0)
        r3, eps3, _, _ = tabular_qlearning(env, policy_file = ("tuning_third.pkl"), episodes_num = 10000, alpha=0.1, gamma= 0.9, eps_decay=0.9, epsilon=1.0)
        plot_hypeparameters(r1, r2, r3)
    else: print("Error")

