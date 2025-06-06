import torch
import torch.nn as nn
import os
import constants as c
import torch.optim as optim
import random, math
import numpy as np
from scipy.special import binom
from collections import deque
from bin.Memory import ReplayMemory, Transition
from bin.Environment import MedicalImageEnvironment
from bin.RLModels.DQN import DQN
from bin.RLModels.A2C import A2C
from bin.RLModels.DDQN import DDQN
from utils.parser import Experiment
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self,  action_dim : int,
                        train_environment: MedicalImageEnvironment = None,
                        eval_environment: MedicalImageEnvironment = None,
                        test_environment: MedicalImageEnvironment = None, 
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
                        evaluation_interval=10):
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
        self.current_episode = 0

        self.best_val_reward = float('-inf')
        self.rl_framework = rl_framework
        if self.rl_framework == "DQN":
            self.rl_model = DQN(action_dim=action_dim, 
                                logger=logger, 
                                gamma=gamma, 
                                model_type=model_type,
                                experiment=self.experiment,
                                tau=tau,
                                lr=lr,
                                max_epsilon=max_epsilon,
                                min_epsilon=min_epsilon,
                                decay=decay,
                                n_actions=self.env.n_actions if task == "train" else self.test_env.n_actions,
                                n_sample_points=self.env.n_sample_points if task == "train" else self.test_env.n_sample_points)
        elif self.rl_framework == "DDQN":
            self.rl_model = DDQN(action_dim=action_dim, 
                                logger=logger, 
                                gamma=gamma, 
                                model_type=model_type,
                                experiment=self.experiment,
                                tau=tau,
                                lr=lr,
                                max_epsilon=max_epsilon,
                                min_epsilon=min_epsilon,
                                decay=decay,
                                n_actions=self.env.n_actions if task == "train" else self.test_env.n_actions,
                                n_sample_points=self.env.n_sample_points if task == "train" else self.test_env.n_sample_points)
        elif self.rl_framework == "A2C":
            self.rl_model = A2C(action_dim=action_dim, 
                                logger=logger, 
                                gamma=gamma, 
                                model_type=model_type,
                                experiment=self.experiment,
                                tau=tau,
                                lr=lr,
                                max_epsilon=max_epsilon,
                                min_epsilon=min_epsilon,
                                decay=decay,
                                n_actions=self.env.n_actions if task == "train" else self.test_env.n_actions,
                                n_sample_points=self.env.n_sample_points if task == "train" else self.test_env.n_sample_points)
        
        self.memory = ReplayMemory(capacity=1000)

        if task != "train":
            assert model_name is not None, "Model named cannot be none"
            
            self.rl_model.load_model(dataLoader=dataLoader, model_name=model_name)
            self.rl_model.eval()
            self.logger.debug(f"Loaded Policy net {model_name}")
        
        self.logger.debug(f"Initialized Trainer with parameters:\n"
                          f"Reinforcement framework {rl_framework}\n"
                          f"Episodes: {episodes}\n"
                          f"Max Steps: {max_steps}\n"
                          f"Decay: {decay}\n"
                          f"Max epsilon: {max_epsilon}\n"
                          f"Min epsilon: {min_epsilon}\n"
                          f"Gamma: {gamma}\n"
                          f"Learning Rate: {lr}\n"
                          f"Model Type: {model_type}\n"
                          f"Experiment: {experiment.name}\n"
                          f"Tau: {tau}")

    def train(self):
        for episode in range(self.episodes):
            self.current_episode = episode
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
                # Get next action
                actions = self.rl_model.select_action(state, location_data, self.total_steps, evaluate=False)
                
                # Return Result of action on environment
                next_state, next_location_data, rewards, done = self.env.step(actions)

                # Format output
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
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

                self.rl_model.optimize_model(self.memory)
                self.rl_model.update_network()

                self.total_steps += 1
                total_reward += rewards.mean(dim=0).item()
                current_distances = self.env.distance_to_truth
                closest_point = np.minimum(closest_point, current_distances)
                furthest_point = np.maximum(furthest_point, current_distances)

            avg_error_mm = self.env.get_curve_error(t_values=np.linspace(0,1, 10), points=self.env._location)
            worst_error_mm = self.env.get_curve_error(t_values=np.array([0.5]), points=self.env._location)
            end_avg_dist = np.mean(current_distances)
            avg_closest_point = np.mean(closest_point)
            avg_furthest_point = np.mean(furthest_point)
            self.logger.info(
                    f"Episode {episode + 1}: Total Reward = {total_reward:.2f} | "
                    f"Final Avg Distance {end_avg_dist:.2f} | Average error in mm {np.round(avg_error_mm,2)} | Worst Error in mm {np.round(worst_error_mm, 2)} "
                    f"Avg Closest Point = {avg_closest_point:.2f} | Avg Furthest Point = {avg_furthest_point:.2f}"
            )
            self.logger.insert_train_row(episode+1, total_reward, end_avg_dist, avg_error_mm, worst_error_mm, avg_closest_point, avg_furthest_point)
                
            if (episode + 1) % self.eval_interval == 0:
                self.logger.info(f"===== Validation Run =====")
                avg_val_reward = self._evaluate(self.eval_env)
                self.logger.info(f"Average Validation Reward: {avg_val_reward:.2f}")
                if avg_val_reward > self.best_val_reward:
                    self.best_val_reward = avg_val_reward
                    self.logger.info(f"New best validation reward! Saving model.")
                    self.rl_model.save_model(dataLoader=self.dataLoader, model_name=f"{self.rl_framework}-{self.model_type}-{self.experiment.name}" + "-best-model")
                
        if self.model_name is not None:
            self.rl_model.save_model(dataLoader=self.dataLoader, model_name=self.model_name)
        else:
            self.rl_model.save_model(dataLoader=self.dataLoader, model_name=f"{self.rl_framework}-{self.model_type}-{self.experiment.name}")

    def _evaluate(self, environment : MedicalImageEnvironment):
        """
        Runs evaluation episodes using the trained policy network without exploration.
        """

        self.rl_model.eval()
        evaluation_total_reward = []
        evaluation_errors_avg = []
        evaluation_errors_worst = []
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

                CPD_distance_mm = environment.get_curve_error(t_values=np.linspace(0, 1, 100), points=environment._location).mean()
                closest_point = np.full(self.agents, float('inf'))
                furthest_point = np.zeros(self.agents)
                total_reward = 0
                found_truth = np.zeros(self.agents, dtype=bool)
                self.total_steps = 0

                while self.total_steps <= self.eval_steps:
                    actions = self.rl_model.select_action(state, location_data, self.total_steps, evaluate=True)
                    
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

                    self.total_steps += 1
                    total_reward += rewards.mean(dim=0).item()
                    current_distances = environment.distance_to_truth
                    closest_point = np.minimum(closest_point, current_distances)
                    furthest_point = np.maximum(furthest_point, current_distances)

                avg_error_mm = environment.get_curve_error(t_values=np.linspace(0, 1, 100), points=environment._location)
                worst_error_mm = environment.get_curve_error(t_values=np.array([0.5]), points=environment._location)
                naive_error_mm = environment.get_curve_error(t_values=np.linspace(0, 1, 100), points=environment.midpoint).mean()
                evaluation_total_reward.append(total_reward)
                
                end_avg_dist = np.mean(current_distances)
                avg_closest_point = np.mean(closest_point)
                avg_furthest_point = np.mean(furthest_point)
                self.logger.info(
                        f"Evaluation Episode {episode + 1}: Total Reward = {total_reward:.2f} | "
                        f"Final Avg Distance {end_avg_dist:.2f} | Average error in mm {np.round(avg_error_mm,2)} | Worst Error in mm {np.round(worst_error_mm, 2)} "
                        f"Avg Closest Point = {avg_closest_point:.2f} | Avg Furthest Point = {avg_furthest_point:.2f}"
                )
                metrics_true, metrics_pred = environment.get_aortic_valve_metrics()
                self.logger.insert_val_row(self.current_episode+1, episode+1, total_reward, end_avg_dist, avg_error_mm, worst_error_mm, avg_closest_point, avg_furthest_point, naive_error_mm, CPD_distance_mm, metrics_true, metrics_pred)
                
                evaluation_errors_avg.append(avg_error_mm)
                evaluation_errors_worst.append(worst_error_mm)

        self.logger.info("===== Evaluation Summary =====")
        make_boxplot(evaluation_errors_avg)
        make_boxplot(evaluation_errors_worst)
        avg_total_reward = np.mean(evaluation_total_reward)
        print(f"Average Total Reward: {avg_total_reward:.2f}")
        self.rl_model.train()  # Return to train mode
        return avg_total_reward

    def test(self):
        self._evaluate(self.test_env)

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
