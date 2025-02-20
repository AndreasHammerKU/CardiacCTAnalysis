
import json
import nibabel as nib
import os

class DataLoader():
    def __init__(self, dataset_dir="", image_dir="images", landmarks_dir="landmarks"):

        self.dataset_dir = dataset_dir
        self.image_dir =  os.path.join(dataset_dir, image_dir)
        self.landmarks_dir = os.path.join(dataset_dir, landmarks_dir)

    def _load_nifti(self, file_path):
        nii_img = nib.load(file_path)
        affine = nii_img.affine  # Get transformation matrix
        return nii_img.get_fdata(), affine

    def load_data(self, image_name):
        # Load NIfTI file
        nifti_path = os.path.join(self.image_dir, image_name + '.nii.gz')
        nifti_data, affine = self._load_nifti(nifti_path)

        # Check scanner orientation 

        # Load landmark JSON
        with open(os.path.join(self.landmarks_dir, image_name + '.json')) as file:
            landmark_data = json.load(file)

        return nifti_data, affine, landmark_data