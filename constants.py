import os

DATASET_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "Dataset", "Data")

IMAGE_FOLDER = os.path.join(DATASET_FOLDER, 'images')

LABELS_FOLDER = os.path.join(DATASET_FOLDER, 'landmarks')

ANCHOR_DATAPOINTS = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']

LABELLED_DATAPOINTS = ['RCI', 'LCI', 'NCI']

MODEL_PATH = os.path.join('.', 'models')
