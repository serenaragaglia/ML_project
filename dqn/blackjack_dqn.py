import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

#hyperparameters
EPISODES_NUM = 100000
ALPHA = 0.1
GAMMA = 0.999
MIN_EPS = 0.1
EPS_DECAY = 0.9999

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
        self.optimizer.zero_grad()
        q_predictions = self.model(states) #forward pass: computes the network output from the batch
        loss = self.loss_fn(q_predictions, q_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def predict_q_value(self, state):
        state_input = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_input)
        return q_values.cpu().numpy()
    
def epsilon_update(epsilon, decay = EPS_DECAY):
    epsilon = max(epsilon* decay, MIN_EPS)
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

def train_blackjack(env, episodes_num = EPISODES_NUM, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0, batch_size = 64):
    state_dim = len(encode(env.reset()[0])) #number of values that compose the state, so its dimension
    num_actions = env.action_space.n #number of actions: hit and stay

    q_network = DQN(state_dim, num_actions, device = "cpu")
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = 1.0
    epsilon_per_episode = []
    tot_rewards = []
    tot_loss = []

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

        mean_loss = np.mean(loss_per_ep) if loss_per_ep else 0.0
        tot_loss.append(mean_loss)

        if (episode + 1)%100 == 0 and tot_rewards:
            mean_reward = np.mean(tot_rewards[-100:])
            print(f"Episode {episode+1} / {episodes_num}, Reward: {mean_reward:.2f}, Epsilon: {epsilon:.3f}")

    return q_network, tot_rewards, epsilon_per_episode, tot_loss

def run_policy(env, q_net, episodes = EPISODES_NUM):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        finished = False
        total_reward = 0

        while not finished:
            q_values = q_net.predict_q_value(encode(state))[0]
            action = int(np.argmax(q_values))
            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
    return rewards

def random_episodes(env, episodes = EPISODES_NUM):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        finished = False
        total_reward = 0

        while not finished:            
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
    return rewards

def save_policy(q_network, filename):
    torch.save(q_network.state_dict(), filename)

def load_policy(state_dim, num_actions, filename):
    q_network = DQN(state_dim, num_actions)
    q_network.load_state_dict(torch.load(filename))
    q_network.eval()
    return q_network

def plot_random_vs_greedy(random_rewards, greedy_rewards, epsilon):
    window = 2000
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

def plot_loss(tot_loss):
    window = 1500
    avg_loss = np.convolve(tot_loss, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12,6))
    plt.plot(avg_loss, color='crimson', linewidth=2)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Mean Loss (smoothed)", fontsize=12)
    plt.title("Loss Trend", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = gym.make('Blackjack-v1', natural=True, sab=False)

    mode = input("Choose the modality : 0 = training, 1 = running policy, 2 = tuning with different hyperparameters  ").strip()
    policy_file = input("File policy name (empty fot default): ").strip()
    if policy_file == "":
        policy_file = "default_dqn.pth"

    if mode == "0":
        q_net, rewards, epsilon, loss = train_blackjack(env, episodes_num = EPISODES_NUM, gamma = GAMMA, eps_decay = EPS_DECAY, epsilon = 1.0, batch_size=64)
        save_policy(q_net, policy_file)
        ran_rewards = random_episodes(env)
        avg_reward = np.mean(rewards)
        print(f"Greedy mean: {np.mean(rewards):.2f}")
        print(f"Random mean: {np.mean(ran_rewards):.2f}")
        plot_random_vs_greedy(ran_rewards, rewards, epsilon)
        plot_loss(loss)
        #plot_training_stats(win, loss, 100)
    elif mode == "1":
        ran_rewards = random_episodes(env)
        state_dim = len(encode(env.reset()[0]))
        num_actions = env.action_space.n
        q_net = load_policy(state_dim, num_actions, policy_file)
        greedy_rewards = run_policy(env, q_net, episodes=EPISODES_NUM)

        print(f"\nMean greedy reward: {np.mean(greedy_rewards):.2f}")
        print(f"Mean random reward: {np.mean(ran_rewards):.2f}")
        #plot_per_policy(ran_rewards, 'Random')
        #plot_per_policy(greedy_rewards, 'Greedy')
    elif mode == "2":
        q1, r1, eps1, l1 = train_blackjack(env, episodes_num = 10000, gamma= 0.99, eps_decay=0.995, epsilon=1.0, batch_size=32)
        save_policy(q1, "first_tuning.pth")
        q2, r2, eps2, l2 = train_blackjack(env,  episodes_num = 10000, gamma= 0.9, eps_decay=0.99, epsilon=1.0, batch_size=64)
        save_policy(q2, "second_tuning.pth")
        q3, r3, eps3, l3 = train_blackjack(env, episodes_num = 10000,  gamma= 0.9, eps_decay=0.9, epsilon=1.0, batch_size=128)
        save_policy(q3, "third_tuning.pth")
        #plot_hypeparameters(r1, r2, r3)
        #plot_hyperparameters_subplots(r1, r2, r3)
    else: print("Error")