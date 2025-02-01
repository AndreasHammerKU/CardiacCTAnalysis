import numpy as np
import os
import utils.visualiser as vis
import json
import constants as c

np.set_printoptions(suppress=True, precision=6)  # Suppress scientific notation, set decimal places
image_name = 'n2'

nifti_data, voxels = vis.load_data(image_name=image_name)

print(voxels)
 
P1 = np.array(voxels['R'])
P2 = np.array(voxels['L'])
P3 = np.array(voxels['N'])


v1 = P2 - P1
v2 = P3 - P1
normal = np.linalg.norm(np.cross(v1, v2)) / (np.cross(v1, v2))

A,B,C = normal

D = -np.dot(normal, voxels['R'])

print("A: {}, B: {}, C: {}, D: {}".format(A,B,C,D))

app = vis.create_slice_viewer(nifti_data,voxels)
app.run_server(debug=True)
