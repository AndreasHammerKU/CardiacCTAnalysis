import numpy as np
import utils.visualiser as vis
import utils.io_utils as io

np.set_printoptions(suppress=True, precision=6)  # Suppress scientific notation, set decimal places
image_name = 'n2'

nifti_data, affine, landmarks = io.load_data(image_name=image_name)

# TODO handle the rest of the points properly: filtering points
landmarks = {k: landmarks[k] for k in ['R', 'RLC', 'RNC'] if k in landmarks}

voxels = vis.world_to_voxel(landmarks=landmarks, affine=affine)
 
P1 = np.array(voxels['R'])
P2 = np.array(voxels['RLC'])
P3 = np.array(voxels['RNC'])


v1 = P2 - P1
v2 = P3 - P1
normal = (np.cross(v2, v1)) / np.linalg.norm(np.cross(v2, v1))
#print(nifti_data.shape)
A,B,C = normal

D = -np.dot(normal, voxels['R'])

target_normal = np.array([1, 0, 0])

rotation = False
print(voxels)
if rotation:
    R = vis.get_rotation_matrix(normal, target_normal)
    nifti_data = vis.rotate_image(nifti_data, R)

    voxels = vis.rotate_landmarks(voxels, R)


app = vis.create_slice_viewer(nifti_data, voxels)
app.run_server(debug=True)
