from gym import spaces
import gym
import numpy as np
from collections import (Counter, defaultdict, deque, namedtuple)
import random
from utils.geometry_fitting import LeafletGeometry
from utils.visualiser import ras_to_lps, world_to_voxel
import copy

class MedicalImageEnvironment(gym.Env):

    def __init__(self, task="train", dataLoader=None, n_sample_points=5, memory_size=28, vision_size=(21, 21, 21), agents=6, image_list=None, logger=None):

        super(MedicalImageEnvironment, self).__init__()

        self.logger = logger
        self.dataLoader = dataLoader
        self.image_list = image_list
        self.agents = agents
        self.task = task
        
        self.actions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        self.n_actions = len(self.actions)
        self.vision_size = vision_size
        
        self.n_sample_points = n_sample_points
        self.t_values = np.linspace(0,1,n_sample_points)

        self.width, self.height, self.depth = vision_size
        # Box must be odd
        assert self.width % 2 == 1 and self.height % 2 == 1 and self.depth % 2 == 1

        self.dims = len(vision_size)
        self.state_size = (self.agents, self.n_sample_points, self.width, self.height, self.depth)

        self.get_next_image()
        self.reset()


    def get_next_image(self):
        image_name = random.choice(self.image_list)
        self.image, affine, landmarks = self.dataLoader.load_data(image_name=image_name)

        landmarks =  ras_to_lps(landmarks)
        voxel_landmarks = world_to_voxel(landmarks=landmarks, affine=affine)
        geometry = LeafletGeometry(voxel_landmarks)
        geometry.calculate_bezier_curves()

        self._p0, _ground_truth, self._p2 = zip(*geometry.Control_points)
        _ground_truth = list(_ground_truth)
        for i in range(self.agents):
            _ground_truth[i] = np.rint(_ground_truth[i])
        
        self._ground_truth = _ground_truth
        self.midpoint = [(self._p0[i] + self._p2[i]) // 2 for i in range(len(self._p0))]
        
        self.logger.debug("Loaded image: {} with ground truth {} and starting point {}".format(image_name, self._ground_truth[0], self.midpoint[0]))

    def reset(self):
        self._location = [(self.midpoint[i][0], self.midpoint[i][1], self.midpoint[i][2]) for i in range(self.agents)]
        self.state = self._update_state()
        return self.state

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

    def step(self, action):
        # Single agents approach
        new_location = np.clip(self._location[0] + np.array(self.actions[action]), [0, 0, 0], np.array(self.image.shape) - 1)

        reward = np.linalg.norm(self._location[0] - self._ground_truth[0]) - np.linalg.norm(new_location - self._ground_truth[0])
        
        self._location[0] = new_location
        self.state = self._update_state()


        done = np.array_equal(self._location[0], self._ground_truth[0])
        return self.state, reward, done

def bezier_curve(p0, p1, p2, t):
    curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, p1) + np.outer(t ** 2, p2)
    return curve.astype(int)  # Convert to integer indices for accessing grid values

        