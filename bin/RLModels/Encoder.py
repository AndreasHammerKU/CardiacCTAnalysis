import torch.nn as nn
import torch


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, n_features):
        super(FeatureEncoder, self).__init__()

        # (batch_size, in_channels, 21, 21, 21)
        self.n_features = n_features
        self.conv0 = self.conv_block(in_channels=in_channels, out_channels=8)
        
        # (batch_size, 8, 10, 10, 10)
        self.conv1 = self.conv_block(in_channels=8, out_channels=16)
       
        # (batch_size, 16, 5, 5, 5)
        self.conv2 = self.conv_block(in_channels=16, out_channels=32)
        
        # (batch_size, 32, 2, 2, 2)
        # view: (batch_size, 256)

    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(-1, self.n_features)

    def conv_block(self, in_channels, out_channels):
        """Convolutional block with two 3D convolutions, batchnorm, and dropout."""
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.PReLU()
        )
        return block