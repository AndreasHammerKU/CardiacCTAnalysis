import torch
from utils.parser import Experiment

# Generic Reinforcement Learning framework
# Defines class methods that each 
# reinforcement learning architectures should implement
class RLModel:
    def __init__(self,  action_dim : int,
                        logger=None, 
                        model_name=None,
                        model_type="Network3D",
                        attention=False,
                        experiment=Experiment.WORK_ALONE,
                        n_sample_points=5,
                        n_actions=6,
                        lr=0.001, 
                        gamma=0.90, 
                        max_epsilon=1.0, 
                        min_epsilon=0.01, 
                        decay=250, 
                        agents=6, 
                        tau=0.005,
                        use_unet=False):
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.n_sample_points = n_sample_points
        self.gamma = gamma
        self.tau = tau
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.agents = agents
        self.lr = lr
        self.use_unet = use_unet
        self.logger = logger
        self.experiment = experiment
        self.attention = attention
    
    def select_action(self, state, location, curr_step, evaluate):
        raise NotImplementedError("This method should be overridden in the subclass.")
    
    def optimize_model(self, transitions):
        raise NotImplementedError("This method should be overridden in the subclass.")
    
    def update_network(self):
        raise NotImplementedError("This method should be overridden in the subclass.")
    
    def eval(self):
        raise NotImplementedError("This method should be overridden in the subclass.")
    
    def train(self):
        raise NotImplementedError("This method should be overridden in the subclass.")