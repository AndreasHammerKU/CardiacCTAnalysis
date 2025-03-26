import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from baseline.BaseEnvironment import MedicalImageEnvironment
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


class BaseUNetTrainer:
    def __init__(self, image_list=None, dataLoader=None, logger=None, init_features=16):
        self.dataLoader = dataLoader
        self.image_list = image_list
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_image_size = 128
        self.init_features = init_features
        self.model_path = f'Unet-{self.init_features}.pth'

        #batch_size = len(self.image_list)
        input_channels = 1
        output_channels = 1
        self.model = UNet3D(in_channels=input_channels, out_channels=output_channels, 
                init_features=self.init_features).to(self.device)

        self.env = MedicalImageEnvironment(
            dataLoader=dataLoader,
            logger=logger,
            task="eval",
            image_list=self.image_list
        )

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def create_distance_fields(self, max_distance=5, granularity=50):
        self.logger.info("Creating Distance Fields")
        for i in tqdm(range(len(self.image_list))):
            image_name = self.image_list[i]
            self.env.get_next_image()
            self.env.reset()
            distance_field = self.env.get_distance_field(max_distance=max_distance, 
                                                         granularity=granularity)

            self.dataLoader.save_distance_field(image_name, distance_field)
    
    def show_DF_from_file(self, image_name, axis=1, slice_index=64):
        distance_field = self.dataLoader.load_distance_field(image_name)
        _show_distance_fields(distance_field, axis=axis, slice_index=slice_index)

    def show_DF_prediction(self, image_name, axis=1, slice_index=64):
        image, _, _ = self.dataLoader.load_data(image_name=image_name)
        true_DF = self.dataLoader.load_distance_field(image_name)
        input_data = torch.tensor(image, dtype=torch.float32, device=self.device)
        print("True_DF: ", true_DF.shape)

        self.model.eval()
        predicted_DF = self.model(input_data.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        print("predict_DF: ", predicted_DF.shape)
        self.model.train()

        _show_distance_fields(predicted_DF.detach().numpy(), axis=axis, slice_index=slice_index)
        _show_distance_fields(true_DF, axis=axis, slice_index=slice_index)


    def load_images(self):
        images = np.zeros((
            len(self.image_list),
            self.fixed_image_size,
            self.fixed_image_size,
            self.fixed_image_size), dtype=np.float32
        )
        self.logger.info("Loading training images")
        for i in tqdm(range(len(self.image_list))):
            image_name = self.image_list[i]
            nifti_data, _, _ = self.dataLoader.load_data(image_name=image_name)
            images[i] = nifti_data
        
        self.train_data = torch.tensor(images, dtype=torch.float32, device=self.device)

        self.logger.info("Loading distance fields")
        distance_fields = np.zeros((
            len(self.image_list),
            self.fixed_image_size,
            self.fixed_image_size,
            self.fixed_image_size), dtype=np.float32
        )
        for i in tqdm(range(len(self.image_list))):
            image_name = self.image_list[i]
            distance_field = self.dataLoader.load_distance_field(image_name=image_name)
            images[i] = distance_field

        self.distance_fields = torch.tensor(distance_fields, dtype=torch.float32, device=self.device)

    def train(self, n_epochs=2):
        self.load_images()
        
        criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(n_epochs):
            self.model.train()
            running_loss = 0.0
            for i in range(len(self.image_list)):
                optimizer.zero_grad()
                # Add singleton batch, channel dimensions and remove them again
                outputs = self.model(self.train_data[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                loss = criterion(outputs, self.distance_fields[i])

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

            self.logger.info(f"Epoch {epoch+1}: loss {running_loss/len(self.image_list)}")
        
        torch.save(self.model.state_dict(), self.model_path)

def _show_distance_fields(distance_field, axis=1, slice_index=64):
    _, ax = plt.subplots(figsize=(6, 6))
    Nx, Ny, Nz = distance_field.shape
    if axis == 0:  # X-plane
        img = distance_field[slice_index, :, :]
        extent = [0, Ny, 0, Nz]
    elif axis == 1:  # Y-plane
        img = distance_field[:, slice_index, :]
        extent = [0, Nx, 0, Nz]
    else:  # Z-plane
        img = distance_field[:, :, slice_index]
        extent = [0, Nx, 0, Ny]
    
    ax.imshow(img.T, origin="lower", cmap="magma", extent=extent, vmin=0, vmax=1)
    #ax.scatter(b_x, b_y, color="cyan", s=10, label="Bezier Samples")
    ax.set_title(f"Distance Field Slice (axis={axis}, index={slice_index})")
    ax.legend()
    plt.colorbar(ax.imshow(img.T, origin="lower", cmap="magma", vmin=0, vmax=1))
    plt.show()

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