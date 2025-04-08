
import json
import nibabel as nib
import os
import constants as c
from collections import namedtuple
from tqdm import tqdm
from typing import Tuple
import numpy as np
import torch
import collections


DataEntry = namedtuple('DataEntry',
                        ('image', 'affine', 'landmarks'))

class DataLoader():
    def __init__(self, dataset_dir="", model_dir=c.MODEL_PATH, image_dir="images", landmarks_dir="landmarks", distance_dir="distance_fields"):

        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.image_dir =  os.path.join(dataset_dir, image_dir)
        self.landmarks_dir = os.path.join(dataset_dir, landmarks_dir)
        self.distance_fields = os.path.join(dataset_dir, distance_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.preloaded = False

    def preload_images(self, image_list):
        self.preloaded_images = {}
        for i in tqdm(range(len(image_list))):
            image_name = image_list[i]
            nifti_data, affine, landmark_data = self.load_data(image_name=image_name)
            self.preloaded_images[image_name] = DataEntry(nifti_data, affine, landmark_data)

        self.preloaded = True

    def _load_nifti(self, file_path):
        nii_img = nib.load(file_path)
        affine = nii_img.affine  # Get transformation matrix
        return nii_img.get_fdata(), affine

    def load_data(self, image_name, trim_image=True):
        if self.preloaded:
            return self.preloaded_images[image_name]
        
        # Load NIfTI file
        nifti_path = os.path.join(self.image_dir, image_name + '.nii.gz')
        nifti_data, affine = self._load_nifti(nifti_path)

        # Load landmark JSON
        with open(os.path.join(self.landmarks_dir, image_name + '.json')) as file:
            landmark_data = json.load(file)

        landmarks =  _map_landmarks(landmark_data, function=_ras_to_lps)
        inv_affine = np.linalg.inv(affine)
        voxel_landmarks = _map_landmarks(landmarks, _world_to_voxel, inv_affine)
        
        if trim_image:
            nifti_data, voxel_landmarks = self._crop_image(image=nifti_data, landmarks=voxel_landmarks)

        return nifti_data, affine, voxel_landmarks
    
    def save_distance_field(self, image_name : str, distance_field : np.ndarray):
        np.savez_compressed(os.path.join(self.distance_fields, f"d_{image_name}.npz"), distance=distance_field)

    def load_distance_field(self, image_name : str) -> np.ndarray:
        loaded_data = np.load(os.path.join(self.distance_fields, f"d_{image_name}.npz"))
        return loaded_data["distance"]
    
    def save_model(self, model_name : str, state_dict : collections.OrderedDict):
        torch.save(state_dict, os.path.join(self.model_dir, f"{model_name}.pt"))

    def load_model(self, model_name : str):
        return torch.load(os.path.join(self.model_dir, f"{model_name}.pt"), map_location=self.device)

    def save_unet(self, model_name : str, state_dict : collections.OrderedDict):
        torch.save(state_dict.state_dict(), os.path.join(self.model_dir, f"{model_name}.pth"))

    def load_unet(self, model_name : str):
        return torch.load(os.path.join(self.model_dir, f"{model_name}.pth"), map_location=self.device)

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

def _move_landmarks(point, offset):
    return (np.array(point) - np.array(offset)).tolist()