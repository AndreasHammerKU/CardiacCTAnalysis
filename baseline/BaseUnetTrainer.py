import torch
import torch.nn as nn
import torch.optim as optim
from baseline.BaseEnvironment import MedicalImageEnvironment
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from baseline.BaseUnet import UNet3D


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
        _show_distance_fields([distance_field], axis=axis, slice_index=slice_index)

    def show_DF_prediction(self, image_name, axis=1, slice_index=64):
        image, _, _ = self.dataLoader.load_data(image_name=image_name)
        true_DF = self.dataLoader.load_distance_field(image_name)
        input_data = torch.tensor(image, dtype=torch.float32, device=self.device)

        self.model.eval()
        predicted_DF = self.model(input_data.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        self.model.train()

        images = [predicted_DF.detach().cpu().numpy(), true_DF]
        _show_distance_fields(images, axis=axis, slice_index=slice_index)

        _, ax = plt.subplots(figsize=(6, 6))
        Nx, Ny, Nz = image.shape
        if axis == 0:  # X-plane
            img = image[slice_index, :, :]
            extent = [0, Ny, 0, Nz]
        elif axis == 1:  # Y-plane
            img = image[:, slice_index, :]
            extent = [0, Nx, 0, Nz]
        else:  # Z-plane
            img = image[:, :, slice_index]
            extent = [0, Nx, 0, Ny]
        ax.imshow(img.T, origin="lower", cmap="magma", extent=extent)
        ax.set_title(f"Image Slice (axis={axis}, index={slice_index})")
        plt.show()


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
        LEARNING_RATE = 3e-4

        #criterion = nn.MSELoss()  # Mean Squared Error loss for regression
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
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

def _show_distance_fields(distance_fields, axis=1, slice_index=64):
    num_fields = len(distance_fields)
    _, axes = plt.subplots(1, num_fields, figsize=(6* num_fields, 6))
    Nx, Ny, Nz = distance_fields[0].shape
    if num_fields == 1:
        axes = [axes]
    for i in range(num_fields):
        distance_field = distance_fields[i]
        ax = axes[i]
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