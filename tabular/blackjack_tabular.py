import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
env = gym.make('Blackjack-v1', natural=False, sab=False);

#hyperparameters
EPISODES_NUM = 10000
ALPHA = 0.1
GAMMA = 0.9
MIN_EPS = 0.01
EPS_DECAY = 0.999

#qtable structure
q_table = defaultdict(lambda : np.zeros(env.action_space.n));

#epsilon greedy policy
def epsilon_greedy(state, epsilon):
    n = random.random()
    if n < epsilon:
        return env.action_space.sample()
    else:
        np.argmax(q_table[state])

def tabular_qlearning(env, q_table, epsilon = 1.0):
    rewards_per_episode = []
    for episode in range(EPISODES_NUM):
        state, _ = env.reset()
        finished = False 
        total_reward = 0

        while not finished:
            action = epsilon_greedy(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            finished = terminated or truncated
            total_reward += reward

            next_action = np.argmax(q_table[state])

            state = next_state

        epsilon = max(MIN_EPS, epsilon * EPS_DECAY)
        rewards_per_episode.append(total_reward)

