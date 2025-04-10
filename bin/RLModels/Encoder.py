import torch.nn as nn
import torch


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, n_features):
        super(FeatureEncoder, self).__init__()

        self.n_features = n_features
        self.conv0 = self.conv_block(in_channels=in_channels, out_channels=8)
        self.maxpool0 = self.max_pool_layer()
        self.conv1 = self.conv_block(in_channels=8, out_channels=16)
        self.maxpool1 = self.max_pool_layer()
        self.conv2 = self.conv_block(in_channels=16, out_channels=32)
        self.maxpool2 = self.max_pool_layer()
        self.conv3 = self.conv_block(in_channels=32, out_channels=64)
        self.maxpool3 = self.max_pool_layer()
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        return x.view(-1, self.n_features)

    def conv_block(self, in_channels, out_channels):
        """Convolutional block with two 3D convolutions, batchnorm, and dropout."""
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
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