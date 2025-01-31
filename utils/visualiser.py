import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_nifti(file_path):
    """Load a NIfTI (.nii.gz) file and return the image data as a NumPy array."""
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def visualize_slices(image_data, axis=0, num_slices=5):
    """Visualize slices from a 3D image along a specified axis."""
    
    # Determine slice indices
    slice_indices = np.linspace(0, image_data.shape[axis] - 1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))
    
    for i, slice_idx in enumerate(slice_indices):
        if axis == 0:
            img_slice = image_data[slice_idx, :, :]
        elif axis == 1:
            img_slice = image_data[:, slice_idx, :]
        else:
            img_slice = image_data[:, :, slice_idx]
        
        axes[i].imshow(img_slice.T, cmap='gray', origin='lower')
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis('off')
    
    plt.show()