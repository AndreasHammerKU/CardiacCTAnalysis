import os

NEW_DATA = '_2025_aortic_valves'

OLD_DATA = 'Dataset'

FIGURE_FOLDER = os.path.join('.', 'figures')

DATASET_FOLDER = os.path.join(os.path.expanduser('~'), 'Desktop', NEW_DATA)

LOGS_FOLDER = os.path.join(os.path.expanduser('~'), 'Desktop', 'logs')

MODEL_FOLDER = 'models'

DQN_LOGS = 'DQN'

DDQN_LOGS = 'DDQN'

TRAIN_LOGS = 'train'

TEST_LOGS = 'test'

TEST_EXTERNAL_LOGS = 'test_external'

ROI_FOLDER = 'roi'

LANDMARKS_FOLDER = 'json_markers_info'

IMAGE_FOLDER = 'nii_convert'

EXTERNAL_DATA = 'External_Hospital'

NORMAL_DATA = 'Normal'

PATHOLOGICAL_DATA = 'Pathology'

LANDMARKS_MASK_FOLDER = 'landmark_masks'

REGISTRATION_LANDMARKS = 'train_landmarks.npy'

REGISTRATION_GROUND_TRUTH = 'train_ground_truths.npy'

ANCHOR_DATAPOINTS = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']

LABELLED_DATAPOINTS = ['RCI', 'LCI', 'NCI']

MODEL_PATH = os.path.join('.', 'models')
