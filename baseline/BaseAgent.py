import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from scipy.special import binom
from collections import deque
from baseline.BaseMemory import ReplayMemory, Transition
from baseline.BaseDQN import Network3D


class DQNAgent:
    def __init__(self,  environment, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, min_epsilon=0.01, decay=0.995, agents=6, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.agents = agents
        self.n_actions = self.env.n_actions
        print(self.device)

        self.policy_net = Network3D(agents=1, 
                      n_sample_points=self.env.n_sample_points, 
                      number_actions=self.env.n_actions).to(self.device)
        self.target_net = Network3D(agents=1, 
                      n_sample_points=self.env.n_sample_points, 
                      number_actions=self.env.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(capacity=1000)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[random.randint(0, self.action_dim - 1)]], device=self.device, dtype=torch.int64)
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)


    def optimize_model(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size=batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0)
        actions = torch.cat([a.unsqueeze(0) for a in batch.action], dim=0)
        rewards = torch.cat([r.unsqueeze(0) for r in batch.reward], dim=0)

        non_done_mask = torch.tensor([not val for val in batch.done], device=self.device, dtype=torch.bool)
        non_done_next_states = torch.cat([s for s,d in zip(batch.next_state, batch.done) if not d])

        state_action_values = self.policy_net(states).view(
            -1, self.agents, self.n_actions).gather(2, actions).squeeze(-1)

        next_states_values = torch.zeros((batch_size, self.agents), device=self.device)

        with torch.no_grad():
            next_states_values[non_done_mask] = self.target_net(non_done_next_states).max(2).values
        expected_state_action_values = (next_states_values * self.gamma) + rewards
        
        #print(state_action_values.shape, expected_state_action_values.shape)
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def train_dqn(self, episodes=50, max_steps=1000, new_image_interval=5):
        
        for episode in range(episodes):
            if (episode) % new_image_interval == 0:
                self.env.get_next_image()
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            self.epsilon = self.initial_epsilon
            total_reward = 0
            done = False
            total_steps = 0
            while not done and total_steps <= max_steps:
                action = self.select_action(state)

                next_state, reward, done = self.env.step(action)
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                
                self.memory.push(state, action, next_state, reward, done)
                
                state = next_state

                self.optimize_model()
                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                
                self.target_net.load_state_dict(target_net_state_dict)
                total_steps += 1
                total_reward += reward



            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Reached Goal {done}")
        torch.save(self.policy_net.state_dict(), "latest-model.pt")

    def eval_dqn(self, episodes=2, steps=200):
        for episode in range(episodes):
            state = self.env.reset()

            for _ in range(steps):
                action = self.select_action(state)

                next_state, _, _ = self.env.step(action)
                state = next_state

            print(f"Episode {episode + 1}: Final state = {state}")
