import torch
import torch.nn as nn
import torch.optim as optim
from utils.parser import Experiment

class Network3D(nn.Module):
    def __init__(self, agents, n_sample_points, number_actions, location_dim=3, xavier=True, attention=False, experiment=Experiment.WORK_ALONE):
        super(Network3D, self).__init__()

        self.experiment = experiment
        self.agents = agents
        self.n_sample_points = n_sample_points

        if self.experiment == Experiment.SHARE_POSITIONS:# Dimension of relative locations (assuming 3D points)
            self.location_fc = nn.Linear(in_features=location_dim*agents, out_features=32)
        elif self.experiment == Experiment.SHARE_PAIRWISE:
            self.location_fc = nn.Linear(in_features=self.agents**2, out_features=32)

        self.conv0 = self.conv_block(in_channels=5, out_channels=16)
        self.maxpool0 = self.max_pool_layer()
        self.conv1 = self.conv_block(in_channels=16, out_channels=32)
        self.maxpool1 = self.max_pool_layer()
        self.conv2 = self.conv_block(in_channels=32, out_channels=64)
        self.maxpool2 = self.max_pool_layer()
        self.conv3 = self.conv_block(in_channels=64, out_channels=64)
        self.maxpool3 = self.max_pool_layer()

        # (64x2x2x2)
        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=512, out_features=256) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        
        # Modify fc2 to accept location input
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=256 + 
                       (32 * (self.experiment != Experiment.WORK_ALONE)), out_features=128) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=128, out_features=number_actions) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if isinstance(module, (nn.Conv3d, nn.Linear)):
                    torch.nn.init.xavier_uniform_(module.weight)
    
    def conv_block(self, in_channels, out_channels):
        """Convolutional block with two 3D convolutions, batchnorm, and dropout."""
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2)
        )
        return block
    
    def max_pool_layer(self):
        block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        return block
    
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
            x = self.conv0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.maxpool3(x)
            x = x.view(-1, 512)
            #print(x.shape)
            # Pass through first FC layer
            x = self.fc1[i](x)
            x = self.prelu4[i](x)
            print(x.shape, location_data.shape)
            if self.experiment != Experiment.WORK_ALONE:
                x = torch.cat([x, location_data], dim=-1)
            #print(x.shape)

            # Pass through modified second FC layer
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
            output.append(x)

        output = torch.stack(output, dim=1)  # Stack along agent dimension
        return output
    
class CommNet(nn.Module):
    def __init__(self, agents, n_sample_points, number_actions, location_dim=3, xavier=True, attention=False, experiment=Experiment.WORK_ALONE):
        super(CommNet, self).__init__()

        self.experiment = experiment
        self.agents = agents
        self.n_sample_points = n_sample_points

        if self.experiment == Experiment.SHARE_POSITIONS:# Dimension of relative locations (assuming 3D points)
            self.location_fc = nn.Linear(in_features=location_dim*agents, out_features=32)
        elif self.experiment == Experiment.SHARE_PAIRWISE:
            self.location_fc = nn.Linear(in_features=self.agents**2, out_features=32)


        self.conv0 = self.conv_block(in_channels=5, out_channels=16)
        self.maxpool0 = self.max_pool_layer()
        self.conv1 = self.conv_block(in_channels=16, out_channels=32)
        self.maxpool1 = self.max_pool_layer()
        self.conv2 = self.conv_block(in_channels=32, out_channels=64)
        self.maxpool2 = self.max_pool_layer()
        self.conv3 = self.conv_block(in_channels=64, out_channels=64)
        self.maxpool3 = self.max_pool_layer()

        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=(512 + (32 * (self.experiment != Experiment.WORK_ALONE))) * 2, out_features=256) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        
        # Modify fc2 to accept location input
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=256 * 2, out_features=128) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=128 * 2, out_features=number_actions) for _ in range(self.agents)])

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
            x = self.conv0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.maxpool3(x)
            x = x.view(-1, 512)
            if self.experiment != Experiment.WORK_ALONE:
                x = torch.cat([x, location_data], dim=-1)
            input2.append(x)
        input2 = torch.stack(input2, dim=1)
        #print("input 2: ", input2.shape)
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
        #print("input 4: ", x.shape)
        
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
    
    def conv_block(self, in_channels, out_channels):
        """Convolutional block with two 3D convolutions, batchnorm, and dropout."""
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2)
        )
        return block
    
    def max_pool_layer(self):
        block = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        return block