import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.special import binom
from collections import deque


class DQNAgent:
    def __init__(self,  environment, state_dim, action_dim, model, lr=0.001, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay=0.995, agents=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.agents = agents
        self.n_actions = self.env.n_actions

        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=1000)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).view(
            -1, self.n_actions).gather(1, actions).squeeze(1)
        next_q_values = self.model(next_states).view(
            -1, self.n_actions).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def train_dqn(self, episodes=50):

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)

                next_state, reward, done = self.env.step(action)

                self.store_experience(state, action, reward, next_state, done)
                self.train()
                state = next_state
                total_reward += reward

                print("Current Location: {} Ground truth: {} Action: {} Reward: {}".format(self.env._location[0], self.env._ground_truth[0], action, reward))

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def eval_dqn(self, episodes=2, steps=200):
        for episode in range(episodes):
            state = self.env.reset()

            for _ in range(steps):
                action = self.select_action(state)

                next_state, _, _ = self.env.step(action)
                state = next_state

            print(f"Episode {episode + 1}: Final state = {state}")
