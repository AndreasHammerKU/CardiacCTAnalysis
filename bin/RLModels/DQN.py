import torch
import torch.nn as nn
import torch.optim as optim
from utils.parser import Experiment
import random
import math
from bin.Memory import Transition
from bin.RLModels.RLModel import RLModel
from bin.RLModels.Encoder import FeatureEncoder

class DQN(RLModel):
    def __init__(self, action_dim, logger=None, model_name=None, model_type="Network3D", attention=False, experiment=Experiment.WORK_ALONE, n_sample_points=5, n_actions=6, lr=0.001, gamma=0.9, max_epsilon=1, min_epsilon=0.01, decay=250, agents=6, tau=0.005):
        super().__init__(action_dim, logger, model_name, model_type, attention, experiment, n_sample_points, n_actions, lr, gamma, max_epsilon, min_epsilon, decay, agents, tau)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                      attention=self.attention,
                      experiment=self.experiment).to(self.device)
            self.target_net = CommNet(agents=6, 
                      n_sample_points=self.n_sample_points, 
                      number_actions=self.n_actions,
                      attention=self.attention,
                      experiment=self.experiment).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            
    def select_action(self, state, location, curr_step, evaluate=False):
        if evaluate:
            return self.policy_net(state, location).squeeze().max(1).indices.view(self.agents, 1)
        sample = random.random()
        eps_threshold = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1 * curr_step / self.decay)
        if sample < eps_threshold:
            return torch.tensor([[random.randint(0, self.action_dim - 1)] for _ in range(self.agents)], device=self.device, dtype=torch.int64)
        with torch.no_grad():
            return self.policy_net(state, location).squeeze().max(1).indices.view(self.agents, 1)

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

    def update_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)

        self.target_net.load_state_dict(target_net_state_dict)

    def eval(self):
        self.policy_net.eval()
    
    def train(self):
        self.policy_net.train()

    def save_model(self, dataLoader, model_name):
        dataLoader.save_model(model_name, self.policy_net.state_dict())

    def load_model(self, dataLoader, model_name):
        self.policy_net.load_state_dict(dataLoader.load_model(model_name))

    

class Network3D(nn.Module):
    def __init__(self, agents, 
                       n_sample_points, 
                       number_actions, 
                       location_dim=3, 
                       xavier=True, 
                       attention=False, 
                       experiment=Experiment.WORK_ALONE):

        super(Network3D, self).__init__()

        self.experiment = experiment
        self.agents = agents
        self.n_sample_points = n_sample_points

        if self.experiment == Experiment.SHARE_POSITIONS:# Dimension of relative locations (assuming 3D points)
            self.location_fc = nn.Linear(in_features=location_dim*agents, out_features=32)
        elif self.experiment == Experiment.SHARE_PAIRWISE:
            self.location_fc = nn.Linear(in_features=self.agents**2, out_features=32)

        n_featues = 256
        self.encoder = FeatureEncoder(in_channels=n_sample_points, n_features=n_featues)
        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=n_featues, out_features=128) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        
        # Modify fc2 to accept location input
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=128 + 
                       (32 * (self.experiment != Experiment.WORK_ALONE)), out_features=64) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=64, out_features=number_actions) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if isinstance(module, (nn.Conv3d, nn.Linear)):
                    torch.nn.init.xavier_uniform_(module.weight)
    
    def forward(self, state, location=None):
        """
        Input:
        - state: (batch_size, agents, n_sample_points, *image_size)
        - location: (batch_size, agents, location_dim)
        
        Output:
        - (batch_size, agents, number_actions)
        """
        batched = len(state.shape) == 6  # Check if batch dimension exists

        if batched:
            batch_size = state.shape[0]
            location_data = location.view(batch_size, -1) if self.experiment != Experiment.WORK_ALONE else None
        else:
            location_data = location.view(1, -1) if self.experiment != Experiment.WORK_ALONE else None
        
        if self.experiment != Experiment.WORK_ALONE:
            location_data = self.location_fc(location_data)

        output = []
        for i in range(self.agents):
            x = state[:, i] if batched else state[i].unsqueeze(0)
            x = self.encoder(x)
            # Pass through first FC layer
            x = self.fc1[i](x)
            x = self.prelu4[i](x)
            if self.experiment != Experiment.WORK_ALONE:
                x = torch.cat([x, location_data], dim=-1)

            # Pass through modified second FC layer
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
            output.append(x)

        output = torch.stack(output, dim=1)  # Stack along agent dimension
        return output
    
class CommNet(nn.Module):
    def __init__(self, agents, 
                       n_sample_points, 
                       number_actions, 
                       location_dim=3, 
                       xavier=True, 
                       attention=False, 
                       experiment=Experiment.WORK_ALONE):
        
        super(CommNet, self).__init__()

        self.experiment = experiment
        self.agents = agents
        self.n_sample_points = n_sample_points

        if self.experiment == Experiment.SHARE_POSITIONS:# Dimension of relative locations (assuming 3D points)
            self.location_fc = nn.Linear(in_features=location_dim*agents, out_features=32)
        elif self.experiment == Experiment.SHARE_PAIRWISE:
            self.location_fc = nn.Linear(in_features=self.agents**2, out_features=32)

        n_features = 256
        self.encoder = FeatureEncoder(in_channels=n_sample_points, n_features=n_features)

        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=(n_features + (32 * (self.experiment != Experiment.WORK_ALONE))) * 2, out_features=128) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        
        # Modify fc2 to accept location input
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=128 * 2, out_features=64) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=64 * 2, out_features=number_actions) for _ in range(self.agents)])

        self.attention = attention
        if self.attention:
                self.comm_att1 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att2 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att3 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
        
        if xavier:
            for module in self.modules():
                if isinstance(module, (nn.Conv3d, nn.Linear)):
                    torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, state, location=None):
        """
        Input:
        - state: (batch_size, agents, n_sample_points, *image_size)
        - location: (batch_size, agents, location_dim)
        
        Output:
        - (batch_size, agents, number_actions)
        """
        batched = len(state.shape) == 6  # Check if batch dimension exists

        if batched:
            batch_size = state.shape[0]
            location_data = location.view(batch_size, -1) if self.experiment != Experiment.WORK_ALONE else None
        else:
            location_data = location.view(1, -1) if self.experiment != Experiment.WORK_ALONE else None
        
        if self.experiment != Experiment.WORK_ALONE:
            location_data = self.location_fc(location_data) 
        
        input2 = []
        for i in range(self.agents):
            x = state[:, i] if batched else state[i].unsqueeze(0)
            x = self.encoder(x)
            if self.experiment != Experiment.WORK_ALONE:
                x = torch.cat([x, location_data], dim=-1)
            input2.append(x)
        input2 = torch.stack(input2, dim=1)
        # Communication layers
        if self.attention:
            comm = torch.cat([torch.sum((input2.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att1[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])
            
        else:
            comm = torch.mean(input2, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        input3 = []
        for i in range(self.agents):
            x = input2[:, i]
            x = self.fc1[i](torch.cat((x, comm[i]), axis=-1))
            input3.append(self.prelu4[i](x))
        input3 = torch.stack(input3, dim=1)

        if self.attention:
            comm = torch.cat([torch.sum((input3.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att2[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input3, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        input4 = []
        for i in range(self.agents):
            x = input3[:, i]
            x = self.fc2[i](torch.cat((x, comm[i]), axis=-1))
            input4.append(self.prelu5[i](x))
        input4 = torch.stack(input4, dim=1)
        
        if self.attention:
            comm = torch.cat([torch.sum((input4.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att3[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input4, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        output = []
        for i in range(self.agents):
            x = input4[:, i]
            
            x = self.fc3[i](torch.cat((x, comm[i]), axis=-1))
            output.append(x)
        output = torch.stack(output, dim=1)
        return output
