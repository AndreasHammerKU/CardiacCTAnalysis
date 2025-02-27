import torch
import torch.nn as nn
import torch.optim as optim
import random, math
import numpy as np
from scipy.special import binom
from collections import deque
from baseline.BaseMemory import ReplayMemory, Transition
from baseline.BaseDQN import Network3D


class DQNAgent:
    def __init__(self,  state_dim, 
                        action_dim,
                        train_environment=None,
                        eval_environment=None,
                        test_environment=None, 
                        logger=None, 
                        task="train", 
                        model_path=None, 
                        lr=0.001, 
                        gamma=0.99, 
                        max_epsilon=1.0, 
                        min_epsilon=0.01, 
                        decay=250, 
                        agents=6, 
                        tau=0.005, 
                        max_steps=1000,
                        evaluation_steps=30,
                        episodes=50,
                        image_interval=1,
                        evaluation_interval=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = train_environment
        self.eval_env = eval_environment
        self.test_env = test_environment
        self.logger = logger
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.agents = agents
        self.n_actions = self.env.n_actions
        self.max_steps = max_steps
        self.episodes = episodes
        self.image_interval = image_interval
        self.eval_interval = evaluation_interval
        self.eval_steps = evaluation_steps

        self.policy_net = Network3D(agents=1, 
                      n_sample_points=self.env.n_sample_points, 
                      number_actions=self.env.n_actions).to(self.device)
        self.target_net = Network3D(agents=1, 
                      n_sample_points=self.env.n_sample_points, 
                      number_actions=self.env.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity=1000)

        if task != "train":
            assert model_path is not None, "Model path cannot be none"
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy_net.eval()
            self.logger.debug(f"Loaded Policy net from {model_path}")

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1 * self.total_steps / self.decay)
        if sample < eps_threshold:
            return torch.tensor([[random.randint(0, self.action_dim - 1)]], device=self.device, dtype=torch.int64)
        with torch.no_grad(): 
            return self.policy_net(state).squeeze(1).max(1).indices.view(1, 1)


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
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def train_dqn(self, new_image_interval=2):
        
        for episode in range(self.episodes):
            if (episode) % new_image_interval == 0:
                self.env.get_next_image()
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
            total_reward = 0
            done = False
            self.total_steps = 0
            closest_point = float('inf')
            furthest_point = 0
            while not done and self.total_steps <= self.max_steps:
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
                self.total_steps += 1
                total_reward += reward.item()
                current_distance = self.env.distance_to_truth
                closest_point = min(closest_point, current_distance)
                furthest_point = max(furthest_point, current_distance)
            
            self.logger.info(f"Episode {episode + 1}: Total Reward = {total_reward:.2f} | Reached Goal {done} | Closest Point = {closest_point:.2f} | Furthest Point = {furthest_point:.2f}")
            if (episode + 1) % self.eval_interval == 0:
                self.logger.info(f"===== Validation Run =====")
                self._evaluate_dqn(self.eval_env)        
        torch.save(self.policy_net.state_dict(), "latest-model.pt")

    # Runs model in evaluation mode on given environment. Can be used for both validation and testing.
    def _evaluate_dqn(self, environment):
        """
        Runs evaluation episodes using the trained policy network without exploration.
        """
        
        self.policy_net.eval()  # Set the network to evaluation mode

        total_rewards = []
        success_count = 0  # Tracks how many times the agent reaches the goal
        closest_distances = []
        furthest_distances = []
        with torch.no_grad():  # No gradient tracking needed for evaluation
            for episode in range(len(environment.image_list)):
                environment.get_next_image()
                state = environment.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device)

                total_reward = 0
                found_truth = False
                self.total_steps = 0
                closest_point = float('inf')
                furthest_point = 0

                while self.total_steps <= self.eval_steps:
                    action = self.policy_net(state).squeeze(1).max(1).indices.view(1, 1)  # Greedy action selection
                    next_state, reward, done = environment.step(action)
                    
                    if done:
                        found_truth = True
                    
                    reward = torch.tensor([reward], device=self.device)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

                    state = next_state
                    total_reward += reward.item()

                    current_distance = environment.distance_to_truth
                    closest_point = min(closest_point, current_distance)
                    furthest_point = max(furthest_point, current_distance)

                    self.total_steps += 1

                total_rewards.append(total_reward)
                closest_distances.append(closest_point)
                furthest_distances.append(furthest_point)

                if found_truth:
                    success_count += 1  # If the agent reaches the goal, count it as a success

                self.logger.info(f"Evaluation Episode {episode + 1}: Total Reward = {total_reward:.2f} | Reached Goal {done} | Closest Point = {closest_point:.2f} | Furthest Point = {furthest_point:.2f}")


        avg_reward = sum(total_rewards) / len(environment.image_list)
        success_rate = success_count / len(environment.image_list) * 100
        avg_closest = sum(closest_distances) / len(environment.image_list)
        avg_furthest = sum(furthest_distances) / len(environment.image_list)
        self.policy_net.train() # Return to train mode
        self.logger.info("===== Evaluation Summary =====")
        self.logger.info(f"Average Reward: {avg_reward:.2f}")
        self.logger.info(f"Success Rate: {success_rate:.2f}%")
        self.logger.info(f"Average Closest Distance: {avg_closest:.2f}")
        self.logger.info(f"Average Furthest Distance: {avg_furthest:.2f}")

    def test_dqn(self):
        self._evaluate_dqn(self.test_env)