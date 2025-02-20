import numpy as np
import utils.visualiser as vis
import utils.io_utils as io
import utils.geometry_fitting as geom
import matplotlib.pyplot as plt
import logging
from baseline.BaseEnvironment import MedicalImageEnvironment

np.set_printoptions(suppress=True, precision=6)  # Suppress scientific notation, set decimal places

def setup_logger(debug=False):

    logger = logging.getLogger("Logger")
    
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("log.txt")
    console_handler = logging.StreamHandler()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.hasHandlers():  # Avoid duplicate handlers in case of re-runs
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

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
    logger = setup_logger(True)

    env = MedicalImageEnvironment(logger=logger, image_list=['n1', 'n2', 'n3'], agents=1)


if __name__ == "__main__":
    main()
