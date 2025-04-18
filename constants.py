import os

NEW_DATA = "_2025_aortic_valves"

OLD_DATA = "Dataset"

DATASET_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", OLD_DATA)

ROI_FOLDER = "roi"

LANDMARKS_FOLDER = "json_markers_info"

IMAGE_FOLDER = "nii_convert"

EXTERNAL_DATA = "External_Hospital"

NORMAL_DATA = "Normal"

PATHOLOGICAL_DATA = "Pathology"

LANDMARKS_MASK_FOLDER = "landmark_masks"

ANCHOR_DATAPOINTS = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']

LABELLED_DATAPOINTS = ['RCI', 'LCI', 'NCI']

MODEL_PATH = os.path.join('.', 'models')
