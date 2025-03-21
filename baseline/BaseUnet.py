import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from baseline.BaseEnvironment import MedicalImageEnvironment
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


class BaseUNetTrainer:
    def __init__(self, image_list=None, dataLoader=None, logger=None):
        self.dataLoader = dataLoader
        self.image_list = image_list
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = MedicalImageEnvironment(
            dataLoader=dataLoader,
            logger=logger,
            task="eval",
            image_list=self.image_list
        )

    def create_distance_fields(self, max_distance=5, granularity=50):
        self.logger.info("Creating Distance Fields")
        for i in tqdm(range(len(self.image_list))):
            image_name = self.image_list[i]
            self.env.get_next_image()
            self.env.reset()
            distance_field = self.env.get_distance_field(max_distance=max_distance, granularity=granularity)

            self.dataLoader.save_distance_field(image_name, distance_field)
    
    def show_distance_fields(self, image_name, axis=0, slice_index=250):
        distance_field = self.dataLoader.load_distance_field(image_name)

        fig, ax = plt.subplots(figsize=(6, 6))
        Nx, Ny, Nz = distance_field.shape
        axis = 1
        slice_index = 247
        if axis == 0:  # X-plane
            img = distance_field[slice_index, :, :]
            extent = [0, Ny, 0, Nz]
        elif axis == 1:  # Y-plane
            img = distance_field[:, slice_index, :]
            extent = [0, Nx, 0, Nz]
        else:  # Z-plane
            img = distance_field[:, :, slice_index]
            extent = [0, Nx, 0, Ny]
        
        ax.imshow(img.T, origin="lower", cmap="magma", extent=extent)
        #ax.scatter(b_x, b_y, color="cyan", s=10, label="Bezier Samples")
        ax.set_title(f"Distance Field Slice (axis={axis}, index={slice_index})")
        ax.legend()
        plt.colorbar(ax.imshow(img.T, origin="lower", cmap="magma"))
        plt.show()

    def load_images(self):
        fixed_image_size = 512
        images = np.zeros((
            len(self.image_list),
            fixed_image_size,
            fixed_image_size,
            fixed_image_size), dtype=np.float32
        )
        self.logger.info("Loading training images")
        for i in tqdm(range(len(self.image_list))):
            image_name = self.image_list[i]
            nifti_data, _, _ = self.dataLoader.load_data(image_name=image_name)
            z_limit = nifti_data.shape[2]
            images[i, :, :, :z_limit] = nifti_data
        
        self.train_data = torch.tensor(images, dtype=torch.float32, device=self.device)

        self.logger.info("Loading distance fields")
        distance_fields = np.zeros((
            len(self.image_list),
            fixed_image_size,
            fixed_image_size,
            fixed_image_size), dtype=np.float32
        )
        for i in tqdm(range(len(self.image_list))):
            image_name = self.image_list[i]
            distance_field = self.dataLoader.load_distance_field(image_name=image_name)
            z_limit = distance_field.shape[2]
            images[i, :, :, :z_limit] = distance_field

        self.distance_fields = torch.tensor(distance_fields, dtype=torch.float32, device=self.device)
        


    def train(self, n_epochs=2, preload_images=False):
        self.load_images()
        batch_size = len(self.image_list)
        input_channels = 1
        output_channels = 1

        model = UNet3D(in_channels=input_channels, out_channels=output_channels, 
                init_features=32, num_levels=4, num_classes=1)
        
        for epoch in range(n_epochs):
            model.train()
            self.logger.info(f"Epoch {epoch+1}")
            running_loss = 0.0

            print(self.train_data.shape)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet3D, self).__init__()

        # Initial feature size
        features = init_features
        
        # Encoder (downsampling path)
        self.enc1 = self.conv_block(in_channels, features)
        self.enc2 = self.conv_block(features, features * 2)
        self.enc3 = self.conv_block(features * 2, features * 4)
        self.enc4 = self.conv_block(features * 4, features * 8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(features * 8, features * 16)

        # Decoder (upsampling path)
        self.upconv4 = self.upconv_block(features * 16, features * 8)
        self.upconv3 = self.upconv_block(features * 8, features * 4)
        self.upconv2 = self.upconv_block(features * 4, features * 2)
        self.upconv1 = self.upconv_block(features * 2, features)
        
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

    def upconv_block(self, in_channels, out_channels):
        """Upsample and apply a convolutional block."""
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder forward pass with skip connections
        upconv4 = self.upconv4(bottleneck)
        upconv4 = torch.cat([upconv4, enc4], dim=1)  # Skip connection

        upconv3 = self.upconv3(upconv4)
        upconv3 = torch.cat([upconv3, enc3], dim=1)  # Skip connection

        upconv2 = self.upconv2(upconv3)
        upconv2 = torch.cat([upconv2, enc2], dim=1)  # Skip connection

        upconv1 = self.upconv1(upconv2)
        upconv1 = torch.cat([upconv1, enc1], dim=1)  # Skip connection

        # Final output
        output = self.final_conv(upconv1)
        output = self.sigmoid(output)
        return output