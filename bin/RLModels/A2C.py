import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.parser import Experiment
import random
import math
from bin.Memory import Transition
from bin.RLModels.RLModel import RLModel
from bin.RLModels.Encoder import FeatureEncoder
from torch.distributions import Categorical

class A2C(RLModel):
    def __init__(self, action_dim, logger=None, model_name=None, model_type="Network3D", attention=False, experiment=Experiment.WORK_ALONE, n_sample_points=5, n_actions=6, lr=0.001, gamma=0.9, max_epsilon=1, min_epsilon=0.01, decay=250, agents=6, tau=0.005):
        super().__init__(action_dim, logger, model_name, model_type, attention, experiment, n_sample_points, n_actions, lr, gamma, max_epsilon, min_epsilon, decay, agents, tau)

        self.agents = agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor =   Actor(agents=agents,
                             n_sample_points=n_sample_points,
                             number_actions=n_actions,
                             experiment=experiment).to(self.device)
        self.critic = Critic(agents=agents,
                             n_sample_points=n_sample_points,
                             experiment=experiment).to(self.device)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, location, curr_step, evaluate):
        policy_dist = self.actor(state, location)
        actions = []
        for i in range(self.agents):
            dist = Categorical(probs=policy_dist[0,i])
            if evaluate:
                action = torch.argmax(policy_dist[0,i], dim=1).item()
            else:
                action = dist.sample()
            actions.append([action])
        return torch.tensor(actions, device=self.device, dtype=torch.int64)

    def optimize_model(self, memory, batch_size=32):
        if len(memory) < batch_size:
            return
        
        transitions = memory.sample(batch_size=batch_size)
        batch = Transition(*zip(*transitions))
        
        states = torch.cat([s.unsqueeze(0) for s in batch.state], dim=0)
        next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state], dim=0)
        actions = torch.cat([a.unsqueeze(0) for a in batch.action], dim=0)
        rewards = torch.cat([r.unsqueeze(0) for r in batch.reward], dim=0)
        dones = torch.cat([d.unsqueeze(0) for d in batch.done], dim=0)
        
        # Compute the average reward across agents for each transition
        rewards += torch.mean(rewards, axis=1).unsqueeze(1).repeat(1, rewards.shape[1])
        # Critic Loss (Value Function Loss)

        locations = torch.cat([l.unsqueeze(0) for l in batch.location], dim=0) if self.experiment != Experiment.WORK_ALONE else None
        next_locations = torch.cat([l.unsqueeze(0) for l in batch.next_location], dim=0) if self.experiment != Experiment.WORK_ALONE else None

        values = self.critic(states)
        next_values = self.critic(next_states)

        # Check if agents is done
        next_values = (1 - dones.squeeze(-1)) * next_values.view(-1, self.agents)

        td_target = rewards + self.gamma * next_values
        td_error = td_target - values.view(-1, self.agents)
        critic_loss = td_error.pow(2).mean()

        # Actor Loss (Policy Gradient Loss)
        action_probs = self.actor(states)

        log_action_probs = torch.log(action_probs.gather(2, actions).squeeze(-1))
        actor_loss = (td_error.detach() * -log_action_probs).mean()
        
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.critic_optim.step()
        self.actor_optim.step()

    def update_network(self):
        pass

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def save_model(self, dataLoader, model_name):
        dataLoader.save_model(model_name + "-actor", self.actor.state_dict())
        dataLoader.save_model(model_name + "-critic", self.critic.state_dict())

    def load_model(self, dataLoader, model_name):
        self.actor.load_state_dict(dataLoader.load_model(model_name + "-actor"))
        self.critic.load_state_dict(dataLoader.load_model(model_name + "-critic"))

class Actor(nn.Module):
    def __init__(self, agents, 
                       n_sample_points, 
                       number_actions,  
                       experiment=Experiment.WORK_ALONE):
        super().__init__()

        self.experiment = experiment
        self.agents = agents
        self.n_sample_points = n_sample_points
        self.n_features = 256
        self.encoder = FeatureEncoder(in_channels=n_sample_points, n_features=self.n_features)

        self.actor_head = nn.ModuleList(
            [nn.Linear(in_features=self.n_features, out_features=number_actions) for _ in range(self.agents)])
        
    def forward(self, state, location=None):

        batched = len(state.shape) == 6

        policies = []
        for i in range(self.agents):
            x = state[:, i] if batched else state[i].unsqueeze(0)
            #print("Before encoder: ", x)
            x = self.encoder(x)
            #print("Before Head: ", x)
            policy = self.actor_head[i](x)
            policy = F.softmax(policy, dim=-1)
            policies.append(policy)

        return torch.stack(policies, dim=1)
            
class Critic(nn.Module):
    def __init__(self, agents, 
                       n_sample_points, 
                       experiment=Experiment.WORK_ALONE):
        super().__init__()

        self.experiment = experiment
        self.agents = agents
        self.n_sample_points = n_sample_points
        self.n_features = 256
        self.encoder = FeatureEncoder(in_channels=n_sample_points, n_features=self.n_features)

        self.critic_head = nn.ModuleList(
            [nn.Linear(in_features=self.n_features, out_features=1) for _ in range(self.agents)])
        
    def forward(self, state, location=None):

        batched = len(state.shape) == 6

        values = []
        for i in range(self.agents):
            x = state[:, i] if batched else state[i].unsqueeze(0)
            x = self.encoder(x)
            value = self.critic_head[i](x)
            values.append(value)
        
        return torch.stack(values, dim=1)