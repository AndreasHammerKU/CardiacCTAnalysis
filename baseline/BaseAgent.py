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
        self.n_actions = self.env.n_actions if task == "train" else self.test_env.n_actions
        self.n_sample_points = self.env.n_sample_points if task == "train" else self.test_env.n_sample_points
        self.max_steps = max_steps
        self.episodes = episodes
        self.image_interval = image_interval
        self.eval_interval = evaluation_interval
        self.eval_steps = evaluation_steps

        self.policy_net = Network3D(agents=6, 
                      n_sample_points=self.n_sample_points, 
                      number_actions=self.n_actions).to(self.device)
        self.target_net = Network3D(agents=6, 
                      n_sample_points=self.n_sample_points, 
                      number_actions=self.n_actions).to(self.device)
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
            return torch.tensor([[random.randint(0, self.action_dim - 1)] for _ in range(self.agents)], device=self.device, dtype=torch.int64)
        with torch.no_grad(): 
            return self.policy_net(state).squeeze().max(1).indices.view(self.agents, 1)


    def optimize_model(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size=batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0)
        actions = torch.cat([a.unsqueeze(0) for a in batch.action], dim=0)
        rewards = torch.cat([r.unsqueeze(0) for r in batch.reward], dim=0)

        # Compute the average reward across agents for each transition
        avg_rewards = rewards.mean(dim=1, keepdim=True)

        non_done_mask = torch.tensor([not any(d) for d in batch.done], device=self.device, dtype=torch.bool)
        non_done_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not any(d)])

        state_action_values = self.policy_net(states).view(
            -1, self.agents, self.n_actions).gather(2, actions).squeeze(-1)

        next_states_values = torch.zeros((batch_size, self.agents), device=self.device)

        with torch.no_grad():
            next_states_values[non_done_mask] = self.target_net(non_done_next_states).max(2).values

        # Bellman equation with averaged rewards
        expected_state_action_values = (next_states_values * self.gamma) + avg_rewards

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
            done = np.zeros(len(self.env._location), dtype=bool)
            self.total_steps = 0
            closest_point = np.full(len(self.env._location), float('inf'))
            furthest_point = np.zeros(len(self.env._location))

            while not np.all(done) and self.total_steps <= self.max_steps:
                
                actions = self.select_action(state)  # Assume select_action returns actions for all agents


                next_state, rewards, done = self.env.step(actions)
                rewards = torch.tensor(rewards, device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

                self.memory.push(state, actions, next_state, rewards, done)

                state = next_state

                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)

                self.target_net.load_state_dict(target_net_state_dict)
                self.total_steps += 1

                total_reward += rewards.mean(dim=0).item()
                current_distances = self.env.distance_to_truth
                closest_point = np.minimum(closest_point, current_distances)
                furthest_point = np.maximum(furthest_point, current_distances)

            self.logger.info(
                    f"Episode {episode + 1}: Total Reward = {total_reward:.2f} | "
                    f"Reached Goal {np.all(done)} | Closest Point = {np.mean(closest_point):.2f} | "
                    f"Furthest Point = {np.mean(furthest_point):.2f}"
            )

            if (episode + 1) % self.eval_interval == 0:
                self.logger.info(f"===== Validation Run =====")
                self._evaluate_dqn(self.eval_env)      

        torch.save(self.policy_net.state_dict(), "latest-model.pt")

    def _evaluate_dqn(self, environment):
        """
        Runs evaluation episodes using the trained policy network without exploration.
        """

        self.policy_net.eval()  # Set the network to evaluation mode

        success_counts = np.zeros(self.agents)  # Track successful goal reaches per agent
        closest_distances = np.full(self.agents, float('inf'))
        furthest_distances = np.zeros(self.agents)

        with torch.no_grad():  # No gradient tracking needed for evaluation
            for episode in range(len(environment.image_list)):
                environment.get_next_image()
                state = environment.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device)

                total_rewards = 0
                found_truth = np.zeros(self.agents, dtype=bool)
                self.total_steps = 0

                while self.total_steps <= self.eval_steps:
                    actions = self.policy_net(state).squeeze().max(1).indices.view(self.agents, 1)  # Greedy action selection
                    next_state, rewards, done = environment.step(actions)
                    found_truth = np.logical_or(found_truth, done.reshape((6)))  # Track if any agent reached the goal
                    rewards = torch.tensor(rewards, device=self.device)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

                    state = next_state
                    total_rewards += rewards.mean(dim=0).item()

                    current_distances = environment.distance_to_truth
                    closest_distances = np.minimum(closest_distances, current_distances)
                    furthest_distances = np.maximum(furthest_distances, current_distances)

                    self.total_steps += 1

                #success_counts += found_truth.astype(int)  # Count successes per agent
                self.logger.info(
                    f"Evaluation Episode {episode + 1}: Total Reward = {total_rewards:.2f} | "
                    f"Reached Goal {found_truth} | Closest Point = {closest_distances} | "
                    f"Furthest Point = {furthest_distances}"
                )

        avg_rewards = total_rewards / len(environment.image_list)
        success_rates = (success_counts / len(environment.image_list)) * 100
        avg_closest = closest_distances.mean()
        avg_furthest = furthest_distances.mean()

        self.policy_net.train()  # Return to train mode
        self.logger.info("===== Evaluation Summary =====")
        self.logger.info(f"Average Closest Distance Across Agents: {avg_closest:.2f}")
        self.logger.info(f"Average Furthest Distance Across Agents: {avg_furthest:.2f}")

    def test_dqn(self):
        self._evaluate_dqn(self.test_env)