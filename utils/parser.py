import argparse
from enum import Enum

class Experiment(Enum):
    WORK_ALONE = 1
    SHARE_POSITIONS = 2
    SHARE_PAIRWISE = 3


    @classmethod
    def from_string(cls, label):
        try:
            return cls[label]
        except KeyError:
            raise argparse.ArgumentTypeError(f"Invalid choice: {label}. Choose from {[e.name for e in cls]}")

class ExperimentConfig:
    def __init__(self, model_type="Network3D", 
                       attention=False, 
                       experiment=Experiment.WORK_ALONE, 
                       lr=0.001, 
                       gamma=0.9, 
                       max_epsilon=1.0,
                       min_epsilon=0.01,
                       decay=250, 
                       agents=6, 
                       tau=0.005, 
                       max_steps=1000,
                       evaluation_steps=30,
                       episodes=50,
                       image_interval=1,
                       evaluation_interval=10):
        self.model_type=model_type
        self.attention=attention
        self.experiment=experiment 
        self.lr=lr
        self.gamma=gamma 
        self.max_epsilon=max_epsilon
        self.min_epsilon=min_epsilon
        self.decay=decay
        self.agents=agents
        self.tau=tau
        self.max_steps=max_steps
        self.evaluation_steps=evaluation_steps
        self.episodes=episodes
        self.image_interval=image_interval
        self.evaluation_interval=evaluation_interval

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
        "-a", "--attention",
        action="store_true",
        help="Enables attention in model"
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

    parser.add_argument(
        "--model_type",
        type=str,
        default="Network3D",
        help="Network Type."
    )

    parser.add_argument(
        "--experiment",
        type=Experiment.from_string,
        choices=list(Experiment),
        required=True,
        help="Choose an experiment type."
    )

    return parser.parse_args()