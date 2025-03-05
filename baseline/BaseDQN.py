import torch
import torch.nn as nn
import torch.optim as optim

class Network3D(nn.Module):
    def __init__(self, agents, n_sample_points, number_actions, location_dim=3, xavier=True):
        super(Network3D, self).__init__()

        self.agents = agents
        self.n_sample_points = n_sample_points
        self.location_dim = location_dim*agents  # Dimension of relative locations (assuming 3D points)

        self.location_fc = nn.Linear(in_features=self.location_dim, out_features=32)

        self.conv0 = nn.Conv3d(
            in_channels=n_sample_points,
            out_channels=16,
            kernel_size=(3, 3, 3),
            padding=2,
            stride=2)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            padding=2)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3, 3),
            padding=2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.prelu2 = nn.PReLU()

        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=864, out_features=128) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        
        # Modify fc2 to accept location input
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=128 + 32, out_features=64) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU() for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=64, out_features=number_actions) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if isinstance(module, (nn.Conv3d, nn.Linear)):
                    torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, state, location):
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
            global_location = location.view(batch_size, -1)
        else:
            global_location = location.view(1, -1)
        
        global_location = self.location_fc(global_location)

        output = []
        for i in range(self.agents):
            x = state[:, i] if batched else state[i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = x.view(-1, 864)

            # Pass through first FC layer
            x = self.fc1[i](x)
            x = self.prelu4[i](x)

            # Concatenate location before second FC layer
            x = torch.cat([x, global_location], dim=-1)
            
            # Pass through modified second FC layer
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
            output.append(x)

        output = torch.stack(output, dim=1)  # Stack along agent dimension
        return output