import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet3D, self).__init__()

        # Initial feature size
        features = init_features
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, features)
        self.enc2 = self.conv_block(features, features * 2)
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.enc4 = self.conv_block(features * 4, features * 8)

        # Max Pool layers
        self.max1 = self.max_pool_layer()
        self.max2 = self.max_pool_layer()
        self.max3 = self.max_pool_layer()
        self.max4 = self.max_pool_layer()
        
        # Bottleneck
        self.bottleneck = self.conv_block(features * 8, features * 16)

        # Upsample Layers
        self.upconv4 = self.upconv_block(features * 16, features * 8)
        self.upconv3 = self.upconv_block(features * 8, features * 4)
        self.upconv2 = self.upconv_block(features * 4, features * 2)
        self.upconv1 = self.upconv_block(features * 2, features)

        # Decoder layers
        self.dec4 = self.conv_block(features * 16, features * 8)
        self.dec3 = self.conv_block(features * 8, features * 4)
        self.dec2 = self.conv_block(features * 4, features * 2)
        self.dec1 = self.conv_block(features * 2, features)
        
        # Final output layer (1x1x1 convolution)
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

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
    
    def upconv_block(self, in_channels, out_channels):
        """Upsample and apply a convolutional block."""
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # [batch, 1, 128, 128, 128]
        enc1 = self.enc1(x)
        # [batch, features, 128, 128, 128]
        max1 = self.max1(enc1)
        # [batch, features, 64, 64, 64]
        
        enc2 = self.enc2(max1)
        # [batch, features * 2, 64, 64, 64]
        max2 = self.max2(enc2)
        # [batch, features * 2, 32, 32, 32]
        
        enc3 = self.enc3(max2)
        # [batch, features * 4, 32, 32, 32]
        max3 = self.max3(enc3)
        # [batch, features * 4, 16, 16, 16]
        
        enc4 = self.enc4(max3)
        # [batch, features * 8, 16, 16, 16]
        max4 = self.max4(enc4)
        # [batch, features * 8, 8, 8, 8]

        bottleneck = self.bottleneck(max4)
        # [batch, features * 16, 8, 8, 8]
        
        # Decoder forward pass with skip connections
        upconv4 = self.upconv4(bottleneck)
        # [batch, features * 8, 16, 16, 16]
        
        upconv4 = torch.cat([upconv4, enc4], dim=1)  # Skip connection
        # [batch, features * 16, 16, 16, 16]
        
        dec4 = self.dec4(upconv4)
        # [batch, features * 8, 16, 16, 16]
        
        upconv3 = self.upconv3(dec4)
        # [batch, features * 4, 32, 32, 32]
        
        upconv3 = torch.cat([upconv3, enc3], dim=1)  # Skip connection
        # [batch, features * 8, 32, 32, 32]
        
        dec3 = self.dec3(upconv3)
        # [batch, features * 4, 32, 32, 32]

        upconv2 = self.upconv2(dec3)
        # [batch, features * 2, 64, 64, 64]
        upconv2 = torch.cat([upconv2, enc2], dim=1)  # Skip connection
        # [batch, features * 4, 64, 64, 64]
        dec2 = self.dec2(upconv2)
        # [batch, features * 2, 64, 64, 64]
        upconv1 = self.upconv1(dec2)
        # [batch, features, 128, 128, 128]
        upconv1 = torch.cat([upconv1, enc1], dim=1)  # Skip connection
        # [batch, features * 2, 128, 128, 128]
        dec1 = self.dec1(upconv1)
        # [batch, features, 128, 128, 128]
        
        # Final output
        output = self.final_conv(dec1)
        # [batch, 1, 128, 128, 128]
        output = self.sigmoid(output)
        # [batch, 1, 128, 128, 128]
        
        return output