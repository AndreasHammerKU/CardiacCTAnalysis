import torch
import torch.nn as nn
import os
import constants as c
import torch.optim as optim
import random, math
import numpy as np
from scipy.special import binom
from collections import deque
from baseline.BaseMemory import ReplayMemory, Transition
from baseline.BaseDQN import DQN
from utils.parser import Experiment
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self,  action_dim : int,
                        train_environment=None,
                        eval_environment=None,
                        test_environment=None, 
                        logger=None,
                        dataLoader=None, 
                        task="train", 
                        model_name=None,
                        model_type="Network3D",
                        attention=False,
                        experiment=Experiment.WORK_ALONE,
                        rl_framework="DQN",
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
                        evaluation_interval=10,
                        use_unet=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = train_environment
        self.experiment = experiment
        self.eval_env = eval_environment
        self.test_env = test_environment
        self.logger = logger
        self.dataLoader = dataLoader
        self.agents = agents
        self.max_steps = max_steps
        self.episodes = episodes
        self.image_interval = image_interval
        self.eval_interval = evaluation_interval
        self.eval_steps = evaluation_steps
        self.model_type = model_type
        self.attention = attention
        self.model_name = model_name

        self.rl_framework = rl_framework
        if self.rl_framework == "DQN":
            self.rl_model = DQN(action_dim=action_dim, 
                                logger=logger, 
                                gamma=gamma, 
                                model_type=model_type,
                                experiment=self.experiment,
                                tau=tau,
                                max_epsilon=max_epsilon,
                                min_epsilon=min_epsilon,
                                decay=decay,
                                n_actions=self.env.n_actions if task == "train" else self.test_env.n_actions,
                                n_sample_points=self.env.n_sample_points if task == "train" else self.test_env.n_sample_points,
                                use_unet=use_unet)
        else:
            pass
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity=1000)

        if task != "train":
            assert model_name is not None, "Model named cannot be none"
            
            self.policy_net.load_state_dict(self.dataLoader.load_model(model_name))
            self.policy_net.eval()
            self.logger.debug(f"Loaded Policy net {model_name}")


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

    def train_dqn(self):
        for episode in range(self.episodes):
            if (episode) % self.image_interval == 0:
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

            errors = self.env.get_curve_error(t_values=np.linspace(0,1, 100))
            self.logger.info(
                    f"Episode {episode + 1}: Total Reward = {total_reward:.2f} | Final Avg Distance {np.mean(current_distances):.2f} | "
                    f"Distances in mm {np.round(errors,2)} | Avg Closest Point = {np.mean(closest_point):.2f} | "
                    f"Avg Furthest Point = {np.mean(furthest_point):.2f}"
            )

            if (episode + 1) % self.eval_interval == 0:
                self.logger.info(f"===== Validation Run =====")
                self._evaluate_dqn(self.eval_env)
                if self.model_name is not None:
                    self.dataLoader.save_model(f"{self.model_name}-episode-{episode+1}", self.policy_net.state_dict())
                else:
                    self.dataLoader.save_model(f"{self.model_type}-{self.experiment.name}-episode-{episode+1}", self.policy_net.state_dict())
        if self.model_name is not None:
            self.dataLoader.save_model(self.model_name, self.policy_net.state_dict())
        else:
            self.dataLoader.save_model(f"{self.model_type}-{self.experiment.name}", self.policy_net.state_dict())

    def _evaluate_dqn(self, environment):
        """
        Runs evaluation episodes using the trained policy network without exploration.
        """

        self.policy_net.eval()  # Set the network to evaluation mode

        evaluation_errors_100 = []
        evaluation_errors_1 = []
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


                errors_100 = environment.get_curve_error(t_values=np.linspace(0,1, 100))
                errors_1 = environment.get_curve_error(t_values=np.array([0.5]))
                
                evaluation_errors_100.append(errors_100)
                evaluation_errors_1.append(errors_1)
                #success_counts += found_truth.astype(int)  # Count successes per agent
                self.logger.info(
                    f"Evaluation Episode {episode + 1}: Total Reward = {total_rewards:.2f} | Final Average Distance = {np.mean(current_distances):.2f} | "
                    f"Error in mm {np.round(errors_1,2)} | Closest Point = {np.round(closest_distances, 2)} | "
                    f"Furthest Point = {np.round(furthest_distances, 2)}"
                )
        make_boxplot(evaluation_errors_100)
        make_boxplot(evaluation_errors_1)

        avg_closest = closest_distances.mean()
        avg_furthest = furthest_distances.mean()

        self.policy_net.train()  # Return to train mode
        self.logger.info("===== Evaluation Summary =====")
        self.logger.info(f"Average Closest Distance Across Agents: {avg_closest:.2f}")
        self.logger.info(f"Average Furthest Distance Across Agents: {avg_furthest:.2f}")

    def test_dqn(self):
        self._evaluate_dqn(self.test_env)

def make_boxplot(error):
    error_data = np.concatenate(error)
    print(f"average error across all agents is {error_data.mean()} mm")

    plt.figure(figsize=(6, 4))
    plt.boxplot(error_data, vert=True, patch_artist=True, showfliers=True)
    # Add titles and labels
    plt.title('Boxplot of Errors of agents to ground truth curve')
    plt.ylabel('Error Value')
    # Show the plot
    plt.show()