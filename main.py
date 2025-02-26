import numpy as np
import utils.io_utils as io
import logging
from baseline.BaseEnvironment import MedicalImageEnvironment
from utils.io_utils import DataLoader
from baseline.BaseAgent import DQNAgent
import constants as c
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(description="DQN Agent Main Script")

    parser.add_argument(
        "-t", "--task",
        choices=["train", "eval", "test"],
        required=True,
        help="Specify the task to run: 'train' for training, 'eval' for evaluation, or 'test' for testing."
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample points (default: 5)."
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Optional path to a pre-trained model file."
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enables debug logs"
    )

    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=1000,
        help="Maximum number of steps per episode (default: 1000)."
    )

    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=100,
        help="Total number of episodes to run (default: 100)."
    )
    
    parser.add_argument(
        "-i", "--image_interval",
        type=int,
        default=1,
        help="Interval for saving images during training (default: 1)."
    )

    parser.add_argument(
        "-p", "--preload",
        action="store_true",
        help="Preload images before running (default: False)."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to a pre-trained model file."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    logger = setup_logger(args.debug)
    if args.dataset is None:
        dataLoader = DataLoader(c.DATASET_FOLDER)
    else:
        dataLoader = DataLoader(args.dataset)
    
    if args.task == "train":
        env = MedicalImageEnvironment(logger=logger, 
                                  dataLoader=dataLoader, 
                                  image_list=['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18', 'n19', 'n20', 'n21', 'n22', 'n23', 'n24', 'n25', 'n26', 'n27', 'n28', 'n29', 'n30'], 
                                  agents=1,
                                  n_sample_points=args.samples,
                                  preload_images=args.preload)
        agent = DQNAgent(environment=env,
                         logger=logger,
                         state_dim=env.state_size,
                         action_dim=env.n_actions,
                         agents=1,
                         model_path=args.model,
                         max_steps=args.steps,
                         episodes=args.episodes,
                         image_interval=args.image_interval)
    
        agent.train_dqn()

    if args.task == "eval":
        env = MedicalImageEnvironment(logger=logger,
                                      task="eval",
                                      dataLoader=dataLoader,
                                      image_list=['n31', 'n32', 'n33', 'n34', 'n35', 'n36', 'n37', 'n38', 'n39', 'n40'],
                                      agents=1,
                                      n_sample_points=args.samples)
        
        agent = DQNAgent(environment=env,
                         task="eval",
                         logger=logger,
                         state_dim=env.state_size,
                         action_dim=env.n_actions,
                         agents=1,
                         model_path=args.model,
                         max_steps=args.steps,
                         episodes=args.episodes)
        
        agent.evaluate_dqn()



if __name__ == "__main__":
    main()
