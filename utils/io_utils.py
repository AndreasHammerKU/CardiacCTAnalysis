
import json
import nibabel as nib
import constants as c
import os

def load_nifti(file_path):
    nii_img = nib.load(file_path)
    affine = nii_img.affine  # Get transformation matrix
    return nii_img.get_fdata(), affine

def load_data(image_name):
    # Load NIfTI file
    nifti_path = os.path.join(c.IMAGE_FOLDER, image_name + '.nii.gz')
    nifti_data, affine = load_nifti(nifti_path)

    # Check scanner orientation 

    # Load landmark JSON
    with open(os.path.join(c.LABELS_FOLDER, image_name + '.json')) as file:
        landmark_data = json.load(file)
    
    return nifti_data, affine, landmark_data