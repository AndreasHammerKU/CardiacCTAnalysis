import torch
import torch.nn as nn
import torch.optim as optim
import random, math
import numpy as np
from scipy.special import binom
from collections import deque
from baseline.BaseMemory import ReplayMemory, Transition
from baseline.BaseDQN import Network3D, CommNet
from utils.parser import Experiment
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self,  state_dim, 
                        action_dim,
                        train_environment=None,
                        eval_environment=None,
                        test_environment=None, 
                        logger=None, 
                        task="train", 
                        model_path=None,
                        model_type="Network3D",
                        attention=False,
                        experiment=Experiment.WORK_ALONE,
                        lr=0.001, 
                        gamma=0.90, 
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
        self.experiment = experiment
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
        self.model_type = model_type
        self.attention = attention
        if model_type == "Network3D":
            self.policy_net = Network3D(agents=6, 
                      n_sample_points=self.n_sample_points, 
                      number_actions=self.n_actions,
                      attention=self.attention,
                      experiment=self.experiment).to(self.device)
            self.target_net = Network3D(agents=6, 
                      n_sample_points=self.n_sample_points, 
                      number_actions=self.n_actions,
                      attention=self.attention,
                      experiment=self.experiment).to(self.device)
        elif model_type == "CommNet":
            self.policy_net = CommNet(agents=6, 
                      n_sample_points=self.n_sample_points, 
                      number_actions=self.n_actions,
                      experiment=self.experiment).to(self.device)
            self.target_net = CommNet(agents=6, 
                      n_sample_points=self.n_sample_points, 
                      number_actions=self.n_actions,
                      experiment=self.experiment).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity=1000)

        if task != "train":
            assert model_path is not None, "Model path cannot be none"
            self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy_net.eval()
            self.logger.debug(f"Loaded Policy net from {model_path}")

    def select_action(self, state, location):
        sample = random.random()
        eps_threshold = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1 * self.total_steps / self.decay)
        if sample < eps_threshold:
            return torch.tensor([[random.randint(0, self.action_dim - 1)] for _ in range(self.agents)], device=self.device, dtype=torch.int64)
        with torch.no_grad():
            return self.policy_net(state, location).squeeze().max(1).indices.view(self.agents, 1)


    def optimize_model(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size=batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0)
        next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state], dim=0)
        actions = torch.cat([a.unsqueeze(0) for a in batch.action], dim=0)
        rewards = torch.cat([r.unsqueeze(0) for r in batch.reward], dim=0)
        dones = torch.cat([d.unsqueeze(0) for d in batch.done], dim=0)

        # Compute the average reward across agents for each transition
        rewards += torch.mean(rewards, axis=1).unsqueeze(1).repeat(1, rewards.shape[1])

        locations = torch.cat([l.unsqueeze(0) for l in batch.location], dim=0) if self.experiment != Experiment.WORK_ALONE else None
        next_locations = torch.cat([l.unsqueeze(0) for l in batch.next_location], dim=0) if self.experiment != Experiment.WORK_ALONE else None
        
        state_action_values = self.policy_net(states, locations).view(
            -1, self.agents, self.n_actions).gather(2, actions).squeeze(-1)

        with torch.no_grad():
            next_states_values = self.target_net(next_states, next_locations).view(-1, self.agents, self.n_actions).max(-1)[0]
        
        next_states_values = (1 - dones.squeeze(-1)) * next_states_values

        # Bellman equation with averaged rewards
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
            
            # Get normalized locations of each agent
            if self.experiment == Experiment.SHARE_POSITIONS:
                location_data = torch.tensor(self.env._location, dtype=torch.float32, device=self.device)
                location_data = torch.abs(location_data - location_data.mean(dim=0, keepdim=True))
            elif self.experiment == Experiment.SHARE_PAIRWISE:
                location_data = torch.tensor(self.env._location, dtype=torch.float32, device=self.device)
                location_data = torch.cdist(location_data, location_data)
            else:
                location_data = None
                next_location_data = None

            total_reward = 0
            done = torch.zeros(self.agents, dtype=torch.int)
            self.total_steps = 0
            closest_point = np.full(len(self.env._location), float('inf'))
            furthest_point = np.zeros(len(self.env._location))

            while not torch.all(done) and self.total_steps <= self.max_steps:
                actions = self.select_action(state, location_data)  # Assume select_action returns actions for all agents

                next_state, next_location_data, rewards, done = self.env.step(actions)

                rewards = torch.tensor(rewards, device=self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                done = torch.tensor(done, dtype=torch.int, device=self.device)
                
                if self.experiment == Experiment.SHARE_POSITIONS:
                    next_location_data = torch.tensor(next_location_data, dtype=torch.float32, device=self.device)
                    next_location_data = torch.abs(next_location_data - next_location_data.mean(dim=0, keepdim=True))
                elif self.experiment == Experiment.SHARE_PAIRWISE:
                    next_location_data = torch.tensor(next_location_data, dtype=torch.float32, device=self.device)
                    next_location_data = torch.cdist(next_location_data, next_location_data)

                self.memory.push(state, location_data, actions, next_state, next_location_data, rewards, done)

                state = next_state

                if self.experiment != Experiment.WORK_ALONE:
                    location_data = next_location_data

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

            errors = self.env.get_curve_error()
            self.logger.info(
                    f"Episode {episode + 1}: Total Reward = {total_reward:.2f} | Final Avg Distance {np.mean(current_distances):.2f} | "
                    f"Distances in mm {np.round(errors,2)} | Avg Closest Point = {np.mean(closest_point):.2f} | "
                    f"Avg Furthest Point = {np.mean(furthest_point):.2f}"
            )

            if (episode + 1) % self.eval_interval == 0:
                self.logger.info(f"===== Validation Run =====")
                self._evaluate_dqn(self.eval_env)      

        torch.save(self.policy_net.state_dict(), f"latest-model-{self.model_type}-{self.experiment.name}.pt")

    def _evaluate_dqn(self, environment):
        """
        Runs evaluation episodes using the trained policy network without exploration.
        """

        self.policy_net.eval()  # Set the network to evaluation mode

        evaluation_errors = []
        with torch.no_grad():  # No gradient tracking needed for evaluation
            for episode in range(len(environment.image_list)):
                environment.get_next_image()
                state = environment.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device)

                if self.experiment == Experiment.SHARE_POSITIONS:
                    location_data = torch.tensor(environment._location, dtype=torch.float32, device=self.device)
                    location_data = torch.abs(location_data - location_data.mean(dim=0, keepdim=True))
                elif self.experiment == Experiment.SHARE_PAIRWISE:
                    location_data = torch.tensor(environment._location, dtype=torch.float32, device=self.device)
                    location_data = torch.cdist(location_data, location_data)
                else:
                    location_data = None
                    next_location_data = None

                closest_distances = np.full(self.agents, float('inf'))
                furthest_distances = np.zeros(self.agents)
                total_rewards = 0
                found_truth = np.zeros(self.agents, dtype=bool)
                self.total_steps = 0

                while self.total_steps <= self.eval_steps:
                    actions = self.policy_net(state, location_data).squeeze().max(1).indices.view(self.agents, 1)  # Greedy action selection
                    
                    next_state, next_location_data, rewards, done = environment.step(actions)
                    
                    found_truth = np.logical_or(found_truth, done.reshape((6)))  # Track if any agent reached the goal
                    
                    
                    rewards = torch.tensor(rewards, device=self.device)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

                    if self.experiment == Experiment.SHARE_POSITIONS:
                        next_location_data = torch.tensor(next_location_data, dtype=torch.float32, device=self.device)
                        next_location_data = torch.abs(next_location_data - next_location_data.mean(dim=0, keepdim=True))
                        location_data = next_location_data
                    elif self.experiment == Experiment.SHARE_PAIRWISE:
                        next_location_data = torch.tensor(next_location_data, dtype=torch.float32, device=self.device)
                        next_location_data = torch.cdist(next_location_data, next_location_data)
                        location_data = next_location_data

                    state = next_state
                    
                    total_rewards += rewards.mean(dim=0).item()

                    current_distances = environment.distance_to_truth
                    closest_distances = np.minimum(closest_distances, current_distances)
                    furthest_distances = np.maximum(furthest_distances, current_distances)

                    self.total_steps += 1

                errors = environment.get_curve_error()

                evaluation_errors.append(errors)
                #success_counts += found_truth.astype(int)  # Count successes per agent
                self.logger.info(
                    f"Evaluation Episode {episode + 1}: Total Reward = {total_rewards:.2f} | Final Average Distance = {np.mean(current_distances):.2f} | "
                    f"Error in mm {np.round(errors,2)} | Closest Point = {np.round(closest_distances, 2)} | "
                    f"Furthest Point = {np.round(furthest_distances, 2)}"
                )
        
        error_data = np.concatenate(evaluation_errors)

        max_bin = 10
        step_size = 0.5
        bins = np.arange(0, max_bin + step_size, step=step_size)
        bins = np.append(bins, np.inf)

        hist, bin_edges = np.histogram(error_data, bins=bins)
        bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-2)]
        bin_labels.append(f"{max_bin}+")

        plt.bar(range(len(hist)), hist, width=0.8, edgecolor="black")
        plt.xticks(range(len(hist)), bin_labels, rotation=45)
        plt.xlabel("Errors in mm")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of curve error in mm for evaulation run")
        plt.show()

        avg_closest = closest_distances.mean()
        avg_furthest = furthest_distances.mean()

        self.policy_net.train()  # Return to train mode
        self.logger.info("===== Evaluation Summary =====")
        self.logger.info(f"Average Closest Distance Across Agents: {avg_closest:.2f}")
        self.logger.info(f"Average Furthest Distance Across Agents: {avg_furthest:.2f}")

    def test_dqn(self):
        self._evaluate_dqn(self.test_env)