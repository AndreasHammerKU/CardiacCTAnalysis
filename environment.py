from gym import spaces
import gym
import numpy as np
from collections import (Counter, defaultdict, deque, namedtuple)
import random
import utils.io_utils as io
import utils.geometry_fitting as geom
import utils.visualiser as vis
import copy

def bezier_curve(p0, p1, p2, t):
    curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, p1) + np.outer(t ** 2, p2)
    return curve.astype(int)  # Convert to integer indices for accessing grid values

Rectangle = namedtuple(
    'Rectangle', [
        'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])

class MedicalImageEnvironment(gym.Env):

    def __init__(self, task="train", n_sample_points=5, memory_size=28, vision_size=(9, 9, 9), agents=6):

        super(MedicalImageEnvironment, self).__init__()

        self.agents = agents
        self.task = task
        self.action_space = spaces.Discrete(6)
        self.action_vectors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        self.actions = self.action_space.n
        self.vision_size = vision_size
        
        self.n_sample_points = n_sample_points
        self.t_values = np.linspace(0,1,n_sample_points)

        self.memory_size = memory_size
        self.width, self.height, self.depth = vision_size
        self.dims = len(vision_size)
        self.rectangle = [[Rectangle(0, 0, 0, 0, 0, 0) for _ in range(self.n_sample_points)] for _ in range(int(self.agents))]

        self.reset_stat()
        self._restart_episode()

    def _restart_episode(self):
        self.terminal = [False] * self.agents
        self.reward = np.zeros((self.agents,))
        self.count = 0
        self.num_runs += 1

        self._loc_history = [
            [(0,) * self.dims for _ in range(self.memory_size)]
            for _ in range(self.agents)]
        self._qvalues_history = [
            [(0,) * self.actions for _ in range(self.memory_size)]
            for _ in range(self.agents)]
        self.current_episode_score = [[]] * self.agents

        self.get_new_run()

    def get_new_run(self):
        self.image_name = 'n{}'.format(random.randint(1,50))
        self._image, affine, landmarks = io.load_data(image_name=self.image_name)
        landmarks = vis.ras_to_lps(landmarks)
        voxel_landmarks = vis.world_to_voxel(landmarks=landmarks, affine=affine)
        geometry = geom.LeafletGeometry(landmarks=voxel_landmarks)
        geometry.calculate_bezier_curves()

        self._p0, self._ground_truth, self._p2 = zip(*geometry.Control_points)
        midpoint = [(self._p0[i] + self._p2[i]) // 2 for i in range(len(self._p0))]
        self._location = [(midpoint[i][0], midpoint[i][1], midpoint[i][2]) for i in range(self.agents)]
        self._start_location = self._location

        self._sample_points = np.zeros((
            self.agents,
            self.n_sample_points,
            len(self.vision_size)
        ))
        for i in range(self.agents):
            self._sample_points[i, :, :] = bezier_curve(self._p0[i], self._location[i], self._p2[i], self.t_values)
        
        self._image_dims = self._image.shape
        self._qvalues = [[0, ] * self.actions] * self.agents
        self._state = self._current_state()

        if self.task == "test":
            self.cur_dist = [0,] * self.agents
        else:
            self.cur_dist = [self.calculate_distance(self._location[i], self._ground_truth[i]) for i in range(self.agents)]
    
    def _current_state(self):
        
        # Define a box of size 'vision_size' around each sample point for each agent
        boxes = np.zeros((
            self.agents,
            self.n_sample_points,
            self.vision_size[0],
            self.vision_size[1],
            self.vision_size[2]
        ), dtype=self._image.dtype)

        for i in range(self.agents):
            for j in range(self.n_sample_points):
                # screen uses coordinate system relative to origin (0, 0, 0)
                screen_xmin, screen_ymin, screen_zmin = 0, 0, 0
                screen_xmax, screen_ymax, screen_zmax = self.vision_size

                # extract boundary locations using coordinate system relative to
                # "global" image
                # width, height, depth in terms of screen coord system


                xmin = np.rint(self._location[i][0]) - round(self.width / 2)
                xmax = np.rint(self._location[i][0]) + round(self.width / 2) + 1
                ymin = np.rint(self._location[i][1]) - round(self.height / 2)
                ymax = np.rint(self._location[i][1]) + round(self.height / 2) + 1
                zmin = np.rint(self._location[i][2]) - round(self.depth / 2)
                zmax = np.rint(self._location[i][2]) + round(self.depth / 2) + 1

                ###########################################################

                # check if they violate image boundary and fix it
                if xmin < 0:
                    xmin = 0
                    screen_xmin = screen_xmax - \
                        len(np.arange(xmin, xmax))
                if ymin < 0:
                    ymin = 0
                    screen_ymin = screen_ymax - \
                        len(np.arange(ymin, ymax))
                if zmin < 0:
                    zmin = 0
                    screen_zmin = screen_zmax - \
                        len(np.arange(zmin, zmax))
                if xmax > self._image_dims[0]:
                    xmax = self._image_dims[0]
                    screen_xmax = screen_xmin + \
                        len(np.arange(xmin, xmax))
                if ymax > self._image_dims[1]:
                    ymax = self._image_dims[1]
                    screen_ymax = screen_ymin + \
                        len(np.arange(ymin, ymax))
                if zmax > self._image_dims[2]:
                    zmax = self._image_dims[2]
                    screen_zmax = screen_zmin + \
                        len(np.arange(zmin, zmax))

                # crop image data to update what network sees
                # image coordinate system becomes screen coordinates
                # scale can be thought of as a stride
                boxes[i, j,
                       screen_xmin:screen_xmax,
                       screen_ymin:screen_ymax,
                       screen_zmin:screen_zmax] = self._image[
                    int(xmin):int(xmax),
                    int(ymin):int(ymax),
                    int(zmin):int(zmax)]

                ###########################################################
                # update rectangle limits from input image coordinates
                # this is what the network sees
                self.rectangle[i][j] = Rectangle(xmin, xmax,
                                              ymin, ymax,
                                              zmin, zmax)
        return boxes

    def step(self, action, q_values, done):
        self._qvalues = q_values
        current_loc = self._location
        next_location = copy.deepcopy(current_loc)

        self.terminal = [False] * self.agents
        go_out = [False] * self.agents

        for i in range(self.agents):
            movement = self.action_vectors[action[i]]
            next_location[i] = current_loc[i] + movement

            if not all(0 <= idx < dim for idx, dim in zip(next_location[i], self._image.shape)):
                next_location[i] = current_loc[i]
                # Left out of bounds
                go_out[i] = True

        if self.task != 'play':
            for i in range(self.agents):
                if go_out[i]:
                    self.reward[i] = -1
                else:
                    self.reward[i] = self._calc_reward(
                        current_loc[i], next_location[i], agent=i)

        self._location = next_location
        self._screen = self._current_state()

        if self.task != 'test':
            for i in range(self.agents):
                self.cur_dist[i] = self.calculate_distance(self._location[i],
                                                     self._ground_truth[i])            

        if self.task == 'train':
            for i in range(self.agents):
                if self.cur_dist[i] <= 1:
                    self.logger.log(f"distance of agent {i} is <= 1")
                    self.terminal[i] = True
                    self.num_success[i] += 1

        self._update_history()
        distance_error = self.cur_dist
        for i in range(self.agents):
            self.current_episode_score[i].append(self.reward[i])

        info = {}
        for i in range(self.agents):
            info[f"score_{i}"] = np.sum(self.current_episode_score[i])
            info[f"gameOver_{i}"] = self.terminal[i]
            info[f"filename_{i}"] = self.image_name
            info[f"agent_xpos_{i}"] = self._location[i][0]
            info[f"agent_ypos_{i}"] = self._location[i][1]
            info[f"agent_zpos_{i}"] = self._location[i][2]
            if self._ground_truth is not None:
                info[f"distError_{i}"] = distance_error[i]
                info[f"landmark_xpos_{i}"] = self._ground_truth[i][0]
                info[f"landmark_ypos_{i}"] = self._ground_truth[i][1]
                info[f"landmark_zpos_{i}"] = self._ground_truth[i][2]
        return self._current_state(), self.reward, self.terminal, info


    def calculate_distance(self, source, target):
        point1 = np.array([source[0], source[1], source[2]])
        point2 = np.array([target[0], target[1], target[2]])

        return np.linalg.norm(point2 - point1)

    def reset_stat(self):
        self.num_runs = 0

    def reset(self):
        self._restart_episode()
        return self._current_state()
    
    def getBestLocation(self):
        ''' get best location with best qvalue from last for locations
        stored in history
        '''
        best_locations = []
        for i in range(self.agents):
            last_qvalues_history = self._qvalues_history[i][-4:]
            last_loc_history = self._loc_history[i][-4:]
            best_qvalues = np.max(last_qvalues_history, axis=1)
            best_idx = best_qvalues.argmin()
            best_locations.append(last_loc_history[best_idx])
        return best_locations

    def _clear_history(self):
        ''' clear history buffer with current states
        '''
        self._loc_history = [
            [(0,) * self.dims for _ in range(self._history_length)]
            for _ in range(self.agents)]
        self._qvalues_history = [
            [(0,) * self.actions for _ in range(self._history_length)]
            for _ in range(self.agents)]
    
    def _update_history(self):
        ''' update history buffer with current states
        '''
        for i in range(self.agents):
            # update location history
            self._loc_history[i].pop(0)
            self._loc_history[i].insert(
                len(self._loc_history[i]), self._location[i])

            # update q-value history
            self._qvalues_history[i].pop(0)
            self._qvalues_history[i].insert(
                len(self._qvalues_history[i]), self._qvalues[i])
            
    def _calc_reward(self, current_loc, next_loc, agent):
        """
        Calculate the new reward based on the decrease in euclidean distance to
        the target location
        """
        curr_dist = self.calculate_distance(current_loc, self._ground_truth[agent])
        next_dist = self.calculate_distance(next_loc, self._ground_truth[agent])
        return curr_dist - next_dist