import gymnasium as gym
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

#hyperparameters
EPISODES_NUM = 10000
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

    def train(self, batch):
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
    
def epsilon_update(epsilon):
    epsilon = max(epsilon* EPS_DECAY, MIN_EPS)
    return epsilon

def encode(state):
    return np.array(state, dtype=np.float32)

def next_action(state, env, epsilon, q_network :DQN ) :
    if random.random() < epsilon:
        return env.action_space.sample()
    q_values = q_network.predict_q_value(encode(state))[0]
    return int(np.argmax(q_values))

def update_model(s, a, r, next_state, done, batch, q_network : DQN):
    training = []
    for(s, a, r, next_state, done) in batch:
        q_current_state = q_network.predict_q_value(encode(s))[0]
        q_next_state = q_network.predict_q_value(encode(next_state))[0]
        if done:
            q_current_state[a] = r
        else:
            q_current_state[a] += ALPHA *(r + GAMMA*np.max(q_next_state) - q_current_state[a])
        training.append((encode(s),  q_current_state))

    return training


