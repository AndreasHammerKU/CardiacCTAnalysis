from gym import spaces
import gym
import numpy as np
from collections import (Counter, defaultdict, deque, namedtuple)
import random
from utils.geometry_fitting import LeafletGeometry
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.special import comb
import plotly.graph_objects as go
from bin.DataLoader import DataLoader
from pycpd import AffineRegistration
import constants as c
import torch
from tqdm import tqdm
import os

class MedicalImageEnvironment(gym.Env):

    def __init__(self, task="train", 
                       dataLoader: DataLoader = None, 
                       n_sample_points=5,
                       vision_size=(21, 21, 21), 
                       agents=6, 
                       image_list=None, 
                       logger=None,
                       train_images=None,
                       trim_image=False):

        super(MedicalImageEnvironment, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.dataLoader = dataLoader
        self.image_list = image_list

        self.agents = agents
        self.task = task
        self.trim_image = trim_image
        
        self.actions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        self.n_actions = len(self.actions)
        self.vision_size = vision_size
        
        self.n_sample_points = n_sample_points
        self.t_values = np.linspace(0,1,n_sample_points+2)[1:-1]

        self.width, self.height, self.depth = vision_size
        # Box must be odd
        assert self.width % 2 == 1 and self.height % 2 == 1 and self.depth % 2 == 1

        self.dims = len(vision_size)
        self.state_size = (self.agents, self.n_sample_points, self.width, self.height, self.depth)

        if task != "train":
            self.current_image = 0
        
        if task != "train":
            if os.path.exists(c.REGISTRATION_LANDMARKS) and os.path.exists(c.REGISTRATION_GROUND_TRUTH):
                self.train_landmarks = np.load(c.REGISTRATION_LANDMARKS)
                self.train_ground_truth = np.load(c.REGISTRATION_GROUND_TRUTH)
            else:
                assert train_images is not None
                self.train_images = train_images
                self._load_train_landmarks()
    
    def _load_train_landmarks(self):
        self.train_landmarks = np.zeros((len(self.train_images), self.agents, 3))
        self.train_ground_truth = np.zeros((len(self.train_images), self.agents, 3))

        for i in tqdm(range(len(self.train_images))):
            _, _, voxel_landmarks = self.dataLoader.load_data(image_name=self.train_images[i], trim_image=self.trim_image)

            geometry = LeafletGeometry(voxel_landmarks)
            geometry.calculate_bezier_curves()

            p0, ground_truthes, _ = zip(*geometry.Control_points)
            self.train_landmarks[i] = np.array(p0)
            self.train_ground_truth[i] = np.array(ground_truthes)

        np.save(c.REGISTRATION_LANDMARKS, self.train_landmarks)
        np.save(c.REGISTRATION_GROUND_TRUTH, self.train_ground_truth)

    def get_next_image(self):
        if self.task == "train":
            image_name = random.choice(self.image_list)
        else:
            image_name = self.image_list[self.current_image]
            self.current_image += 1
            if self.current_image >= len(self.image_list):
                self.current_image = 0
        
        self.image, self.affine, voxel_landmarks = self.dataLoader.load_data(image_name=image_name, trim_image=self.trim_image)
        
        self.geometry = LeafletGeometry(voxel_landmarks)
        self.geometry.calculate_bezier_curves()

        self._p0, _ground_truth, self._p2 = zip(*self.geometry.Control_points)
        self._ground_truth = np.array(_ground_truth, dtype=np.int16)

        if self.task != "train":
            self.inferred_gt = self._get_test_starting_point(np.array(self._p0), self.train_landmarks, self.train_ground_truth)
        
        self.midpoint = [(self._p0[i] + self._p2[i]) // 2 for i in range(self.agents)]

        self.distance_to_middle = np.linalg.norm(self.midpoint - self._ground_truth, axis=1)

    def _get_test_starting_point(self, test_landmarks, train_landmarks, train_ground_truths):
        """
        test_anchors: (6, 3)
        train_anchors: (n_images, 6, 3)
        train_ground_truths: (n_images, 6, 3)
        """
        n_images = train_landmarks.shape[0]
        inferred_landmarks = []
        registration_errors = []
    
        for i in range(n_images):
            train_anchor = train_landmarks[i]   # (6, 3)
            train_gt = train_ground_truths[i] # (6, 3)
    
            reg = AffineRegistration(X=test_landmarks, Y=train_anchor)
            TY, (B, t) = reg.register()
    
            # Transform the ground truth landmarks
            inferred_gt = (train_gt @ B) + t  # (6, 3)
            inferred_landmarks.append(inferred_gt)
    
            # Calculate registration error (simple L2 norm between matched anchors)
            error = np.linalg.norm(test_landmarks - TY, axis=1).mean()
            registration_errors.append(error)
    
        # Convert to arrays
        inferred_landmarks = np.stack(inferred_landmarks)  # (n_images, 6, 3)
        registration_errors = np.array(registration_errors)  # (n_images,)
    
        # Choose the best match (lowest error)
        best_idx = np.argmin(registration_errors)
        best_inferred_gt = inferred_landmarks[best_idx]
    
        return best_inferred_gt

    def reset(self):
        self._get_next_episode()
        return self.state

    def _get_next_episode(self):
        # If in eval mode infer starting points
        if self.task != "train":
            self._location = np.array(self.inferred_gt, dtype=np.int32)
        else:
            self._location = np.array([(self.midpoint[i][0], self.midpoint[i][1], self.midpoint[i][2]) for i in range(self.agents)], dtype=np.int32)
        self.state = self._update_state()

    def _update_state(self):
        self._sample_points = np.zeros((
            self.agents,
            self.n_sample_points,
            self.dims
        ), dtype=np.int16)
        for i in range(self.agents):
            self._sample_points[i, :, :] = bezier_curve(self._p0[i], self._location[i], self._p2[i], self.t_values)
        half_width, half_height, half_depth = self.width // 2, self.height // 2, self.depth // 2 
        boxes = np.zeros((self.agents, 
                               self.n_sample_points, 
                               self.width, 
                               self.height, 
                               self.depth), dtype=self.image.dtype)
        for i in range(self.agents):
            for o, (x,y,z) in enumerate(self._sample_points[i]):
                # Compute valid min/max coordinates within array bounds
                x_min, x_max = max(x - half_width, 0), min(x + half_width + 1, self.image.shape[0])
                y_min, y_max = max(y - half_height, 0), min(y + half_height + 1, self.image.shape[1])
                z_min, z_max = max(z - half_depth, 0), min(z + half_depth + 1, self.image.shape[2])

                # Compute corresponding indices in the zero-padded array
                pad_x_min, pad_x_max = half_width - (x - x_min), half_width + (x_max - x)
                pad_y_min, pad_y_max = half_height - (y - y_min), half_height + (y_max - y)
                pad_z_min, pad_z_max = half_depth - (z - z_min), half_depth + (z_max - z)
    
                # Copy the valid region from the image to the preallocated box
                boxes[i, o, pad_x_min:pad_x_max, pad_y_min:pad_y_max, pad_z_min:pad_z_max] = \
                self.image[x_min:x_max, y_min:y_max, z_min:z_max]
        return boxes

    def step(self, actions):
        # Multi-agent approach
        action_array = np.array([self.actions[a] for a in actions])
        new_locations = np.clip(
            self._location + action_array,
            [0, 0, 0],
            np.array(self.image.shape) - 1
        )
        
        rewards = np.linalg.norm(self._location - self._ground_truth, axis=1) - np.linalg.norm(new_locations - self._ground_truth, axis=1)
        
        self._location = new_locations
        self.state = self._update_state()
        
        self.distance_to_truth = np.linalg.norm(self._location - self._ground_truth, axis=1)
        done = np.all(self._location == self._ground_truth, axis=1, keepdims=True)

        return self.state, self._location, rewards, done
    
    def visualize_current_state(self, granularity=50):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        # Ground truth points
        for point in self._p0:
            ax.scatter(point[0], point[1], point[2], color='black', marker='o')
        for point in self._p2:
            ax.scatter(point[0], point[1], point[2], color='black', marker='o')
        for point in self._ground_truth:
            ax.scatter(point[0], point[1], point[2], color='black', marker='o')

        for point in self._location:
            ax.scatter(point[0], point[1], point[2], color='red', marker='o')

        t_values = np.linspace(0, 1, granularity)
        true_bezier_curves = [plotting_bezier_curve(self._p0[i], self._ground_truth[i], self._p2[i], t_values) for i in range(self.agents)]

        current_bezier_cruves = [plotting_bezier_curve(self._p0[i], self._location[i], self._p2[i], t_values) for i in range(self.agents)]
        
        for curve in true_bezier_curves:
            ax.plot(curve[:,0], curve[:,1], curve[:,2], color='green')

        for curve in current_bezier_cruves:
            ax.plot(curve[:,0], curve[:,1], curve[:,2], color='red')

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("3D Scatter Plot")

        # Show plot
        plt.show()

    def get_curve_error(self, t_values):
        p0_world = np.array([np.dot(self.affine, np.append(point, 1))[:3] for point in self._p0])
        ground_truth_world = np.array([np.dot(self.affine, np.append(point, 1))[:3] for point in self._ground_truth])
        p2_world = np.array([np.dot(self.affine, np.append(point, 1))[:3] for point in self._p2])
        location_world = np.array([np.dot(self.affine, np.append(point, 1))[:3] for point in self._location])

        error = np.zeros(self.agents, dtype=np.float32)
        for i in range(self.agents):
            predicted_curve = plotting_bezier_curve(p0_world[i], location_world[i], p2_world[i], t_values)

            true_curve = plotting_bezier_curve(p0_world[i], ground_truth_world[i], p2_world[i], t_values)

            error[i] = np.sum(np.abs(predicted_curve - true_curve)) / len(predicted_curve)

        return error      
    
    def get_distance_field(self, max_distance=5, granularity=50):
        bezier_points = np.zeros((
            self.agents * granularity,
            self.dims
        ), dtype=np.int16)
        t_values = np.linspace(0, 1, granularity)
        for i in range(self.agents):
            offset = i*granularity
            bezier_points[offset:offset+granularity, :] = bezier_curve(self._p0[i], self._ground_truth[i], self._p2[i], t_values)

        return self.compute_distance_field(bezier_samples=bezier_points, max_distance=max_distance)

    # Compute distance field
    def compute_distance_field(self, bezier_samples, max_distance):
        Nx, Ny, Nz = self.image.shape

        # Generate 3D grid points
        x = np.arange(0, Nx)
        y = np.arange(0, Ny)
        z = np.arange(0, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # (Nx*Ny*Nz, 3)

        # Build KD-tree for fast nearest-neighbor lookup
        tree = KDTree(bezier_samples)

        # Find nearest Bézier sample for each grid point
        distances, _ = tree.query(grid_points)

        # Normalize distances to [0, 1]
        distances = np.where(distances <= max_distance, 1, 0)

        # Reshape back to 3D grid
        return distances.reshape(Nx, Ny, Nz)
    
    def get_bounding_box(self, padding=20):
        points = self._p0 + self._p2
        min_x = int(min(p[0] for p in points))
        max_x = int(max(p[0] for p in points))
        min_y = int(min(p[1] for p in points))
        max_y = int(max(p[1] for p in points))
        min_z = int(min(p[2] for p in points))
        max_z = int(max(p[2] for p in points))

        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        min_z = max(0, min_z - padding)
        max_x = min(self.image.shape[0], max_x + padding)
        max_y = min(self.image.shape[1], max_y + padding)
        max_z = min(self.image.shape[2], max_z + padding)

        return np.array([min_x, max_x, min_y, max_y, min_z, max_z], dtype=np.int16)


def bezier_curve(p0, p1, p2, t):
    curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, p1) + np.outer(t ** 2, p2)
    return curve.astype(int)  # Convert to integer indices for accessing grid values

def plotting_bezier_curve(p0, p1, p2, t):
    curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, p1) + np.outer(t ** 2, p2)
    return curve  # Convert to integer indices for accessing grid values

