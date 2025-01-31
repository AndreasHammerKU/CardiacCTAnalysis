import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import os

def load_nifti(file_path):
    nii_img = nib.load(file_path)
    return nii_img.get_fdata()

def downsample_volume(volume, scale=0.25):
    from scipy.ndimage import zoom
    return zoom(volume, scale, order=1)  # Linear interpolation

def create_3d_visualization(nifti_data, max_points=50000):
    volume = (nifti_data - np.min(nifti_data)) / (np.max(nifti_data) - np.min(nifti_data))
    volume = (volume * 255).astype(np.uint8)
    
    x, y, z = np.where(volume > np.percentile(volume, 99))  # Select top intensity points
    values = volume[x, y, z]
    
    if len(x) > max_points:
        indices = np.random.choice(len(x), size=max_points, replace=False)
        x, y, z, values = x[indices], y[indices], z[indices], values[indices]
    
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,  # Reduce size for performance
            color=values,
            colorscale='gray',
            opacity=0.3
        )
    ))
    
    fig.update_layout(title='Interactive 3D NIfTI Visualization',
                      scene=dict(
                          xaxis_title='X Axis',
                          yaxis_title='Y Axis',
                          zaxis_title='Z Axis'))
    
    fig.show()

dataset_folder = '/mnt/c/Users/Andre/Desktop/Dataset/Data/'
file_path = os.path.join(dataset_folder, 'images', 'n1.nii.gz') 
nifti_data = load_nifti(file_path)
create_3d_visualization(nifti_data)
