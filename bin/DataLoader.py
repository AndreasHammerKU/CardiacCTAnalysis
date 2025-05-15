
import json
import nibabel as nib
from nibabel.orientations import aff2axcodes
import matplotlib.pyplot as plt
import os
import constants as c
from collections import namedtuple
from tqdm import tqdm
from typing import Tuple
import numpy as np
import torch
import collections
import glob
import random

DataEntry = namedtuple('DataEntry',
                        ('image', 'affine', 'landmarks'))

class DataLoader():
    def __init__(self, dataset_dir="", logger=None, model_dir=c.MODEL_PATH, include_pathological=True, seed=None):
        
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.logger = logger
        self.seed = seed
        
        image_dir = os.path.join(dataset_dir, c.ROI_FOLDER, c.IMAGE_FOLDER)
        normal_images = glob.glob(os.path.join(image_dir, c.NORMAL_DATA, '*'))
        pathological_images = glob.glob(os.path.join(image_dir, c.PATHOLOGICAL_DATA, '*'))
        external_images = glob.glob(os.path.join(image_dir, c.EXTERNAL_DATA, '*'))
        self.logger.debug(
            f"Loaded {len(normal_images)} normal images | "
            f"Loaded {len(pathological_images)} pathological images | "
            f"Loaded {len(external_images)} external images"
        )

        landmark_dir = os.path.join(dataset_dir, c.LANDMARKS_FOLDER)
        normal_landmarks = glob.glob(os.path.join(landmark_dir, c.NORMAL_DATA, '*'))
        pathological_landmarks = glob.glob(os.path.join(landmark_dir, c.PATHOLOGICAL_DATA, '*'))
        external_landmarks = glob.glob(os.path.join(landmark_dir, c.EXTERNAL_DATA, '*'))
        self.logger.debug(
            f"Loaded {len(normal_landmarks)} normal landmarks | "
            f"Loaded {len(pathological_landmarks)} pathological landmarks | "
            f"Loaded {len(external_landmarks)} external landmarks"
        )
    
        normal_images_dict, normal_landmarks_dict, normal_lookup = self.create_lookup_dicts(normal_images, normal_landmarks)
        pathological_images_dict, pathological_landmarks_dict, pathological_lookup = self.create_lookup_dicts(pathological_images, pathological_landmarks)
        external_images_dict, external_landmarks_dict, external_lookup = self.create_lookup_dicts(external_images, external_landmarks)

        self.image_dict = normal_images_dict | pathological_images_dict | external_images_dict
        self.landmark_dict = normal_landmarks_dict | pathological_landmarks_dict | external_landmarks_dict

        p_train = 0.7
        p_val = 0.15
        p_test = 0.15
        if include_pathological:
            self.train, self.val, self.test = _split_dataset(normal_lookup + pathological_lookup, p1=p_train, p2=p_val, p3=p_test, seed=self.seed)
        else:
            self.train, self.val, self.test = _split_dataset(normal_lookup, p1=p_train, p2=p_val, p3=p_test, seed=self.seed)

        self.logger.debug(f"Train split {self.train} with {len(self.train)} entries")
        self.logger.debug(f"Validation split {self.val} with {len(self.val)} entries")
        self.logger.debug(f"Test split {self.test} with {len(self.test)} entries")

        self.test_external = external_lookup

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_lookup_dicts(self, image_paths, landmark_paths):
        image_dict = {os.path.splitext(os.path.basename(p))[0].replace('.nii', ''): p for p in image_paths}
        landmark_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in landmark_paths}

        common_keys = set(image_dict.keys()) & set(landmark_dict.keys())

        return image_dict, landmark_dict, sorted(list(common_keys))

    def _load_nifti(self, file_path):
        nii_img = nib.load(file_path)
        affine = nii_img.affine  # Get transformation matrix
        #self.logger.debug(f"Orientation {nib.orientations.aff2axcodes(affine)} | Image shape {nii_img.get_fdata().shape} | ")
        return nii_img.get_fdata(), affine

    def load_data(self, image_name, trim_image=False):
        self.logger.debug(f"Loading image {image_name}")
        # Load NIfTI file
        nifti_path = self.image_dict[image_name]
        nifti_data, affine = self._load_nifti(nifti_path)
        
        # Load landmark JSON
        with open(self.landmark_dict[image_name]) as file:
            landmark_data = json.load(file)

        landmarks =  _map_landmarks(landmark_data, function=_ras_to_lps)

        return nifti_data, affine, landmarks
    
    def save_distance_field(self, image_name : str, distance_field : np.ndarray):
        np.savez_compressed(os.path.join(self.distance_fields, f"d_{image_name}.npz"), distance=distance_field)

    def load_distance_field(self, image_name : str) -> np.ndarray:
        loaded_data = np.load(os.path.join(self.distance_fields, f"d_{image_name}.npz"))
        return loaded_data["distance"]
    
    def save_model(self, model_name : str, state_dict : collections.OrderedDict):
        torch.save(state_dict, os.path.join(self.model_dir, f"{model_name}.pt"))

    def load_model(self, model_name : str):
        return torch.load(os.path.join(self.model_dir, f"{model_name}.pt"), map_location=self.device)

    def _crop_image(self, image: np.ndarray, landmarks: dict, crop_size=(128, 128, 128)) -> Tuple[np.ndarray, dict]: 
        points = []
        for key, value in landmarks.items():
            if key in c.ANCHOR_DATAPOINTS:
                points.append(value)
        np_points = np.array(points)

        centroid = np.mean(np_points, axis=0).astype(int)

        crop_half = np.array(crop_size) // 2
        min_coords = centroid - crop_half
        max_coords = min_coords + np.array(crop_size)

        min_valid = np.maximum(min_coords, 0)
        max_valid = np.minimum(max_coords, image.shape)
        
        crop = np.zeros(crop_size, dtype=image.dtype)

        # Compute where to place the valid extracted region in the zero-padded crop
        insert_min = (min_valid - min_coords).clip(min=0)  # Position in padded crop
        insert_max = insert_min + (max_valid - min_valid)  # End position in padded crop

        # Extract valid region and place it in the padded crop
        crop[insert_min[0]:insert_max[0], insert_min[1]:insert_max[1], insert_min[2]:insert_max[2]] = \
            image[min_valid[0]:max_valid[0], min_valid[1]:max_valid[1], min_valid[2]:max_valid[2]]

        # Offset represents where the crop starts in the original image
        offset = tuple(min_coords)
        landmarks = _map_landmarks(landmarks, _move_landmarks, offset)
        return crop, landmarks

def world_to_voxel(landmarks, inv_affine):
    return _map_landmarks(landmarks, _world_to_voxel, inv_affine)

def voxel_to_world(landmarks, affine):
    return _map_landmarks(landmarks, _voxel_to_world, affine)

def _map_landmarks(dict: dict, function, *args, **kwargs):
    converted_dict = {}
    for key, value in dict.items():
        if isinstance(value[0], (list, tuple, np.ndarray)):  # Check if it's a list of points (curve)
            converted_dict[key] = [function(point,  *args, **kwargs) for point in value]
        else:
            converted_dict[key] = function(value,  *args, **kwargs)
    return converted_dict

def _ras_to_lps(point):
    return np.array([-point[0], -point[1], point[2]]).tolist()

def _world_to_voxel(point, inv_affine):
    return np.dot(inv_affine, np.append(point, 1))[:3].tolist()

def _voxel_to_world(point, affine):
    return np.dot(affine, np.append(point,1))[:3].tolist()

def _move_landmarks(point, offset):
    return (np.array(point) - np.array(offset)).tolist()

def _split_dataset(data, p1=0.7, p2=0.15, p3=0.15, seed=1):
    """
    Splits a list into three parts based on given percentages.

    Args:
        data (list): List to split.
        p1 (float): Percentage for the first split (e.g., train).
        p2 (float): Percentage for the second split (e.g., val).
        p3 (float): Percentage for the third split (e.g., test).
        seed (int, optional): Seed for shuffling.

    Returns:
        tuple: Three lists (split1, split2, split3)
    """
    assert abs(p1 + p2 + p3 - 1.0) < 1e-6, "Percentages must sum to 1.0"

    rng = random.Random(seed)

    data_shuffled = data.copy()
    rng.shuffle(data_shuffled)

    total = len(data)
    n1 = int(total * p1)
    n2 = int(total * p2)

    split1 = data_shuffled[:n1]
    split2 = data_shuffled[n1:n1 + n2]
    split3 = data_shuffled[n1 + n2:]

    return split1, split2, split3