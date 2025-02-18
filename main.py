import numpy as np
import utils.visualiser as vis
import utils.io_utils as io
from nibabel.orientations import aff2axcodes
import utils.geometry_fitting as geom
import matplotlib.pyplot as plt
import environment as env
from trainer import Trainer
from logger import Logger

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
    image, affine, landmarks = io.load_data(image_name=image_name)

    landmarks = vis.ras_to_lps(landmarks)
    print("Image shape: {}".format(image.shape))
    orientation = aff2axcodes(affine)
    print("Image Orientation is: ", orientation)
    voxel_landmarks = vis.world_to_voxel(landmarks=landmarks, affine=affine)
    #landmarks = vis.ras_to_lps(landmarks)
    geometry = geom.LeafletGeometry(voxel_landmarks)
    geometry.calculate_bezier_curves()

    print("Average error in transformed image {} is {}".format(image_name, geometry.get_average_mse()))
    geometry.plot(plot_label_points=True, plot_control_points=True)
    geometry = geom.LeafletGeometry(landmarks=landmarks)
    geometry.calculate_bezier_curves()

    print("Average error in rps image {} is {}".format(image_name, geometry.get_average_mse()))

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
    vision_size = (9,9,9)
    logger = Logger('./logs/', True, 1000, '')
    environ = env.MedicalImageEnvironment(vision_size=vision_size)

    trainer = Trainer(env=environ, image_size=vision_size, logger=logger, init_memory_size=100, replay_buffer_size=100)
    trainer.train()
    


if __name__ == "__main__":
    main()
