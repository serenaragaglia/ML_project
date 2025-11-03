import gymnasium as gym
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import os

#hyperparameters
EPISODES_NUM = 50000
ALPHA = 0.05
GAMMA = 0.999
MIN_EPS = 0.05
EPS_DECAY = 0.999

#https://gymnasium.farama.org/v0.26.3/tutorials/blackjack_tutorial/

window = 1000

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

    win_rate_tot = []
    loss_rate_tot = []
    draw_rate_tot = []


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

        epsilon_per_episode.append(epsilon)
        epsilon = max(MIN_EPS, epsilon * eps_decay)

        if(episode + 1 ) % 100 == 0 : 
            avg_reward = np.mean(rewards_per_episode[-100:])
            win_rate = np.mean([r > 0 for r in rewards_per_episode]) * 100 #we are in the natural envirnment so we can also have a reward of 1.5
            lose_rate = np.mean([r == -1 for r in rewards_per_episode]) * 100
            draw_rate = np.mean([r == 0 for r in rewards_per_episode]) * 100

            win_rate_tot.append(win_rate)
            loss_rate_tot.append(lose_rate)
            draw_rate_tot.append(draw_rate)

            print(f"Ep {episode+1:6d} | AvgReward: {avg_reward:+.3f} | "
              f"WinRate: {win_rate:.1f}% | LoseRate: {lose_rate:.1f}% |  DrawRate: {draw_rate:.1f}%|  Eps: {epsilon:.3f}")
     
    save_policy(policy_file, q_table)

    return np.array(rewards_per_episode), np.array(epsilon_per_episode), win_rate_tot, loss_rate_tot, draw_rate_tot

def save_policy(policy_file, q_table):
    directory = "tabular_policies"
    os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, policy_file)

    with open(filepath, "wb") as f:
        pickle.dump(dict(q_table), f)

def load_policy(env, policy_file):
    directory = "tabular_policies"
    filepath = os.path.join(directory, policy_file)

    with open(filepath, "rb") as f:
        return defaultdict(lambda: np.zeros(env.action_space.n),pickle.load(f))

#run optimal policy
def run_policy(env, policy_file):
    q = load_policy(env, policy_file)

    actions_per_ep = []
    rewards = []

    for _ in range (EPISODES_NUM):
        state, _ = env.reset()
        finished = False
        tot_reward = 0
        actions = []

        while not finished:
            action = np.argmax(q[state])
            actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            tot_reward += reward
        
        rewards.append(tot_reward)
        actions_per_ep.append(actions)
    print(f"Average Reward for optimal policy: {np.mean(rewards):.2f}")

    return rewards, actions_per_ep

#pick random actions
def random_episodes(env):
    rewards = []
    actions_per_ep = []
    for _ in range (EPISODES_NUM):
        state, _ = env.reset()
        finished = False
        tot_reward = 0
        actions = []

        while not finished:
            action = env.action_space.sample()
            actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            tot_reward += reward

        rewards.append(tot_reward)
        actions_per_ep.append(actions)

    return rewards, actions_per_ep

def plot_random_vs_greedy(random_rewards, greedy_rewards, epsilon):
    avg_greedy = np.convolve(greedy_rewards, np.ones(window)/window, mode='valid')
    avg_random = np.convolve(random_rewards, np.ones(window)/window, mode='valid') 

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(avg_greedy, label=f'Average Greedy (Q-learning) Reward', color='royalblue', linewidth=2)
    ax1.plot(avg_random, label=f'Average Random Reward', color='darkorange', linestyle='--', linewidth=2)

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

def plot_policy_reward(rewards, epsilons):

    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    episodes = np.arange(len(smoothed_rewards))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(episodes, smoothed_rewards, color='royalblue', label='Moving Avg Reward')
    ax1.axhline(y=np.mean(rewards), color='red', linestyle='--', label=f'Mean Reward ({np.mean(rewards):.2f})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(epsilons)), epsilons, color='green', label='Epsilon (Exploration)', linestyle='--')
    ax2.set_ylabel('Epsilon', color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')

    fig.suptitle('Learning Progress & Epsilon Decay', fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()

def plot_training_stats(win_rate_history, lose_rate_history, draw_rate_history):
    episodes = np.arange(len(win_rate_history)) * 100  

    plt.figure(figsize=(12,6))
    plt.plot(episodes, win_rate_history, label="Win rate", color="green", linewidth=2)
    plt.plot(episodes, lose_rate_history, label="Loss rate", color="red", linewidth=2)
    plt.plot(episodes, draw_rate_history, label="Draw rate",color="deepskyblue", linewidth=2)

    plt.title("Evolution of games outcomes during training ", fontsize=14)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_per_policy(rewards, policy):
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    std = np.std(rewards)
    episodes = np.arange(len(smoothed))

    plt.figure(figsize=(12,6))
    plt.plot(episodes, smoothed, label=f'{policy} (moving avg)', color='royalblue')
    plt.fill_between(episodes, smoothed-std, smoothed+std, color='blue', alpha=0.1)
    plt.axhline(y=np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    plt.title(f'Reward Trend - {policy}', fontsize=14, fontweight='bold')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_reward_action_trend(actions_per_episode, policy):

    num_episodes = len(actions_per_episode)
    
    hit_freqs = []
    stick_freqs = []
    episode_labels = []
    
    for i in range(0, num_episodes, window):
        batch = actions_per_episode[i:i+window]
        flat_actions = [a for ep in batch for a in ep] #list with all actions within window episodes
        if flat_actions:
            hit_freq = np.mean([1 if a==1 else 0 for a in flat_actions])
        else:
            hit_freq = 0
        hit_freqs.append(hit_freq)
        stick_freqs.append(1 - hit_freq)
        episode_labels.append(i + window//2)
    
    plt.figure(figsize=(12,6))
    plt.plot(episode_labels, hit_freqs, label='Hit (1)', color='dodgerblue', marker='o', markersize=4)
    plt.plot(episode_labels, stick_freqs, label='Stick (0)', color='midnightblue', marker='o', markersize=4)
    
    plt.xlabel('Episode')
    plt.ylabel('Proportion of Actions')
    plt.title(f'Distribution of Hit and Stick Actions Over Episodes for {policy} policy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_hypeparameters(res1, res2, res3):
    step = 10  

    avg1 = np.convolve(res1, np.ones(window)/window, mode='valid')[::step]
    avg2 = np.convolve(res2, np.ones(window)/window, mode='valid')[::step]
    avg3 = np.convolve(res3, np.ones(window)/window, mode='valid')[::step]

    plt.figure(figsize=(12,6))
    plt.plot(avg1, color='royalblue', label="First", linewidth=2, alpha=0.8)
    plt.plot(avg2, color='red', label="Second", linewidth=2, alpha=0.8)
    plt.plot(avg3, color='lime', label="Third", linewidth=2, alpha=0.8)

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Trend for different hyperparameters')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.show()

def plot_hyperparameters_subplots(res1, res2, res3):
    results = [res1, res2, res3]
    titles = ['First', 'Second', 'Third']
    colors = ['royalblue', 'red', 'lime']
    step = 10

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    
    for i, ax in enumerate(axes):
        avg = np.convolve(results[i], np.ones(window)/window, mode='valid')[::step]
        mean_reward = np.mean(results[i])

        ax.plot(avg, color=colors[i], linewidth=2, label='Smoothed Reward')
        ax.axhline(mean_reward, color='gray', linestyle='--', linewidth=1.5, label=f'Mean = {mean_reward:.2f}')
        
        ax.set_title(f'{titles[i]} hyperparameters')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel('Reward')
        ax.legend()

    axes[-1].set_xlabel('Episodes')
    fig.suptitle('Reward trends per hyperparameter set', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()

if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=True, sab=False)

    mode = input("Choose the modality : 0 = training, 1 = running policy, 2 = tuning with different hyperparameters  ").strip()
    policy_file = input("File policy name (empty for default): ").strip()
    if policy_file == "":
        policy_file = "default.pkl"

    if mode == "0":
        rewards, epsilon, win, loss, draws = tabular_qlearning(env, policy_file, episodes_num = EPISODES_NUM, alpha = ALPHA, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0)
        
        ran_rewards = random_episodes(env)
        avg_reward = np.mean(rewards)

        print(f"Greedy mean: {np.mean(rewards):.2f}")
        print(f"Wins mean: {np.mean(win):.2f}")
        print(f"Draws mean: {np.mean(draws):.2f}")
        print(f"Losses mean: {np.mean(loss):.2f}")
        print(f"Random mean: {np.mean(ran_rewards):.2f}")

        plot_random_vs_greedy(ran_rewards, rewards, epsilon)
        plot_training_stats(win, loss, draws)
        plot_policy_reward(rewards, epsilon)

    elif mode == "1":
        ran_rewards, ran_actions = random_episodes(env)
        greedy_rewards, optimal_actions = run_policy(env, policy_file)

        #plot_per_policy(ran_rewards, 'Random')
        #plot_per_policy(greedy_rewards, 'Greedy')

        plot_reward_action_trend(ran_actions, 'Random')
        plot_reward_action_trend(optimal_actions,'Greedy')
    elif mode == "2":
        r1, eps1, _, _, _ = tabular_qlearning(env, policy_file = ("tuning_first.pkl"), episodes_num = 30000, alpha=0.05, gamma= 0.9999, eps_decay=0.9995, epsilon=1.0)
        r2, eps2, _, _, _ = tabular_qlearning(env, policy_file = ("tuning_second.pkl"), episodes_num = 30000, alpha=0.05, gamma= 0.999, eps_decay=0.999, epsilon=1.0)
        r3, eps3, _, _, _ = tabular_qlearning(env, policy_file = ("tuning_third.pkl"), episodes_num = 30000, alpha=0.05, gamma= 0.99, eps_decay=0.9995, epsilon=1.0)
        plot_hypeparameters(r1, r2, r3)
        plot_hyperparameters_subplots(r1, r2, r3)
    else: print("Error")

