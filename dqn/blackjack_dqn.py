import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

#hyperparameters
EPISODES_NUM = 100000
ALPHA = 0.001
GAMMA = 0.999
MIN_EPS = 0.05
EPS_DECAY = 0.999

window = 1000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen  = capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQN(nn.Module):
    def __init__(self, state_dim, num_actions, device="cpu"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        ).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)

    def train_on_batch(self, batch):
        states = torch.tensor(np.array([s for s, q in batch ]), dtype= torch.float32).to(self.device)  #conversion of states from vector to tensor
        q_target = torch.tensor(np.array([q for s, q in batch ]), dtype= torch.float32).to(self.device)  #conversion of target values from vector to tensor
        self.optimizer.zero_grad()  #set to zero all prevoius gradients
        q_predictions = self.model(states) #forward pass: computes the network output from the batch
        loss = self.loss_fn(q_predictions, q_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_q_value(self, state):
        state_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) #add new dimension to tensor in position 0
        with torch.no_grad():
            q_values = self.model(state_input)
        return q_values.cpu().numpy()
    
def epsilon_update(epsilon, decay = EPS_DECAY):
    epsilon = max(MIN_EPS, epsilon* decay)
    return epsilon

def encode(state):
    return np.array(state, dtype=np.float32)

def next_action(state, env, epsilon, q_network :DQN ) :
    n = random.random()
    if n < epsilon:
        return env.action_space.sample()
    else:
        q_values = q_network.predict_q_value(encode(state))[0]
        action = int(np.argmax(q_values))
        return action

def update_model(s, a, r, next_state, done, batch, q_network : DQN, gamma = GAMMA):
    training = []
    for(s, a, r, next_state, done) in batch:
        q_current_state = q_network.predict_q_value(encode(s))[0]  #current state prediction
        q_next_state = q_network.predict_q_value(encode(next_state))[0] #next state prediction

        #computing the target
        if done:
            q_current_state[a] = r
        else:
            q_current_state[a] = r + gamma*np.max(q_next_state)
        training.append((encode(s),  q_current_state))

    return training

def train_blackjack(env, episodes_num = EPISODES_NUM, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0, batch_size = 32):
    state_dim = len(encode(env.reset()[0])) #number of values that compose the state, so its dimension
    num_actions = env.action_space.n #number of actions: hit and stay

    q_network = DQN(state_dim, num_actions, device = "cpu")
    replay_buffer = ReplayBuffer(capacity=100000)

    epsilon_per_episode = []
    tot_rewards = []
    tot_loss = []

    win_rate_tot= []
    loss_rate_tot = []
    draw_rate_tot = []

    for episode in range(episodes_num):
        state, _ = env.reset()
        finished = False
        reward_per_ep = 0
        loss_per_ep = []

        while not finished:
            action = next_action(state, env, epsilon, q_network)

            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            reward_per_ep += reward

            replay_buffer.add(state, action, reward, next_state, finished) #save inside replay buffer

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                training_data = update_model(state, action, reward, next_state, finished, batch, q_network, gamma)
                loss = q_network.train_on_batch(training_data)
                loss_per_ep.append(loss)

            state = next_state   

        epsilon_per_episode.append(epsilon)
        epsilon = epsilon_update(epsilon, eps_decay)

        tot_rewards.append(reward_per_ep)

        mean_loss = np.mean(loss_per_ep) if loss_per_ep else 0.0 #if the loss is not null then compute the mean
        tot_loss.append(mean_loss)

        if (episode + 1)%100 == 0 and tot_rewards:
            mean_reward = np.mean(tot_rewards[-100:])
            win_rate = np.mean([r > 0 for r in tot_rewards]) * 100
            lose_rate = np.mean([r == -1 for r in tot_rewards]) * 100
            draw_rate = np.mean([r == 0 for r in tot_rewards]) * 100

            win_rate_tot.append(win_rate)
            loss_rate_tot.append(lose_rate)
            draw_rate_tot.append(draw_rate)
            
            print(f"Ep {episode+1:6d} | AvgReward: {mean_reward:+.3f} | "
              f"WinRate: {win_rate:.1f}% | LoseRate: {lose_rate:.1f}% | DrawRate: {draw_rate:.1f}% | Eps: {epsilon:.3f}")

    return q_network, tot_rewards, epsilon_per_episode, tot_loss, win_rate_tot, loss_rate_tot, draw_rate_tot

def run_policy(env, q_net, episodes = EPISODES_NUM):
    rewards = []
    actions_per_ep = []

    for episode in range(episodes):
        state, _ = env.reset()
        finished = False
        total_reward = 0
        actions = []

        while not finished:
            q_values = q_net.predict_q_value(encode(state))[0]
            action = int(np.argmax(q_values))
            actions.append(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        actions_per_ep.append(actions)
    return rewards, actions_per_ep

def random_episodes(env):
    rewards = []
    actions_per_ep = []

    for episode in range(EPISODES_NUM):
        state, _ = env.reset()
        finished = False
        total_reward = 0
        actions = []

        while not finished:            
            action = env.action_space.sample()
            actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        actions_per_ep.append(actions)

    return rewards, actions_per_ep

def save_policy(q_network, filename, folder = "policies"):
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)

    torch.save(q_network.state_dict(), filepath)

def load_policy(state_dim, num_actions, filename, folder = "policies"):
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    q_network = DQN(state_dim, num_actions)
    q_network.load_state_dict(torch.load(filepath))
    q_network.eval()

    return q_network

def plot_random_vs_greedy(random_rewards, greedy_rewards, epsilon):
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

def plot_policy_reward(rewards, epsilons):

    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    episodes = np.arange(len(smoothed_rewards))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(episodes, smoothed_rewards, color='royalblue', label='Moving Avg Reward')
    ax1.axhline(y=np.mean(rewards), color='red', linestyle='--', label=f'Mean Reward ({np.mean(rewards):.2f})')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(epsilons)), epsilons, color='green', label='Epsilon (Exploration)', linestyle='--')
    ax2.set_ylabel('Epsilon', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')

    fig.suptitle('Learning Progress & Epsilon Decay', fontsize=14, fontweight='bold')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()

def plot_loss(tot_loss):
    avg_loss = np.convolve(tot_loss, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12,6))
    plt.plot(avg_loss, color='crimson', linewidth=2)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Mean Loss (smoothed)", fontsize=12)
    plt.title("Loss Trend", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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
    policy_file = input("File policy name (empty fot default): ").strip()
    if policy_file == "":
        policy_file = "default_dqn.pth"

    if mode == "0":
        q_net, rewards, epsilon, loss, wins, losses, draws = train_blackjack(env, episodes_num = EPISODES_NUM, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0, batch_size=32)
        save_policy(q_net, policy_file)
        ran_rewards, _ = random_episodes(env)
        avg_reward = np.mean(rewards)

        print(f"Greedy mean: {np.mean(rewards):.2f}")
        print(f"Random mean: {np.mean(ran_rewards):.2f}")
        print(f"Wins mean: {np.mean(wins):.2f}")
        print(f"Draws mean: {np.mean(draws):.2f}")
        print(f"Losses mean: {np.mean(losses):.2f}")

        plot_random_vs_greedy(ran_rewards, rewards, epsilon)
        plot_loss(loss)

        plot_training_stats(wins, losses, draws)

        plot_policy_reward(rewards, epsilon)
    elif mode == "1":
        ran_rewards, ran_actions = random_episodes(env)

        state_dim = len(encode(env.reset()[0]))
        num_actions = env.action_space.n

        q_net = load_policy(state_dim, num_actions, policy_file)
        greedy_rewards, optimal_actions = run_policy(env, q_net, episodes=EPISODES_NUM)

        print(f"\nMean greedy reward: {np.mean(greedy_rewards):.2f}")
        print(f"Mean random reward: {np.mean(ran_rewards):.2f}")

        plot_per_policy(ran_rewards, 'Random')
        plot_per_policy(greedy_rewards, 'Greedy')

        plot_reward_action_trend(ran_actions, 'Random')
        plot_reward_action_trend(optimal_actions, 'Greedy')
    elif mode == "2":
        q1, r1, eps1, l1, _, _, _ = train_blackjack(env, episodes_num = 30000, gamma= 0.9999, eps_decay=0.9995, epsilon=1.0, batch_size=64)
        save_policy(q1, "first_tuning.pth")
        q2, r2, eps2, l2, _, _, _ = train_blackjack(env,  episodes_num = 30000, gamma= 0.999, eps_decay=0.999, epsilon=1.0, batch_size=64)
        save_policy(q2, "second_tuning.pth")
        q3, r3, eps3, l3, _, _, _ = train_blackjack(env, episodes_num = 30000,  gamma= 0.99, eps_decay=0.9995, epsilon=1.0, batch_size=64)
        save_policy(q3, "third_tuning.pth")
        plot_hypeparameters(r1, r2, r3)
        plot_hyperparameters_subplots(r1, r2, r3)
    else: print("Error")