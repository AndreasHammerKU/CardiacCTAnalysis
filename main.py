import numpy as np
import utils.visualiser as vis
import utils.io_utils as io
from nibabel.orientations import aff2axcodes
import utils.geometry_fitting as geom
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=6)  # Suppress scientific notation, set decimal places


def create_slice_viewer(image_name):
    nifti_data, affine, landmarks = io.load_data(image_name=image_name)

    orientation = aff2axcodes(affine)
    print("Image Orientation is: ", orientation)

    landmarks = vis.ras_to_lps(landmarks)

    voxels = vis.world_to_voxel(landmarks=landmarks, affine=affine)

    point_filter = ['R', 'RLC', 'RNC']

    rotated_image, rotated_voxels = vis.align_surface(nifti_data, voxels, point_filter)


    # Project RCI onto plane.
    y_level = rotated_voxels['R'][1]

    curve = np.array(rotated_voxels['RCI'])

    X = curve[:,0]
    Y = curve[:,1]
    Z = curve[:,2]

    new_points = []
    for point in rotated_voxels['RCI']:
        new_point = [point[0], y_level, point[2]]
        new_points.append(new_point)
    rotated_voxels['RCI'] = new_points

    app = vis.create_slice_viewer(rotated_image, rotated_voxels)
    app.run_server(debug=True)

def view_curve(image_name):
    _, affine, landmarks = io.load_data(image_name=image_name)

    orientation = aff2axcodes(affine)
    print("Image Orientation is: ", orientation)

    #landmarks = vis.ras_to_lps(landmarks)
    geometry = geom.LeafletGeometry(landmarks)
    geometry.calculate_bezier_curves()
    control_point = geometry.Control_points[0][1]
    print(control_point)
    inv_affine = np.linalg.inv(affine)
    voxel_point = np.dot(inv_affine, np.append(control_point, 1))[:3].tolist()
    print(voxel_point)


    print("Average error in image {} is {}".format(image_name, geometry.get_average_mse()))
    geometry.plot(plot_label_points=True, plot_control_points=True)

def generate_ground_truth():
    for i in range(50):
        image_name = f'n{i+1}'



def test_points(image_name):
    _, affine, landmarks = io.load_data(image_name=image_name)
    closest = [0, 0, 0]
    for i, point in enumerate(landmarks['RCI']):
        diff = np.array(point) - np.array(landmarks['R'])
        if i == 0:
            closest = diff
        elif np.linalg.norm(closest) > np.linalg.norm(diff):
            closest = diff
    print("For image name {} cloest point {}, norm {}".format(image_name, closest, np.linalg.norm(closest)))

def main():
    #create_slice_viewer()
    #for i in range(10):
    #    image_name = f'n{i+1}'
    #    view_curve(image_name)

    view_curve('n2')
    #create_slice_viewer(image_name=image_name)
    


if __name__ == "__main__":
    main()
