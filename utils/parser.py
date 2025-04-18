import argparse
from enum import Enum
import yaml

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
                       rl_framework="DQN",
                       n_sample_points=5,
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
        self.rl_framework=rl_framework
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
        self.n_sample_points=n_sample_points

    def to_dict(self):
        return {
            'model_type': self.model_type,
            'attention': self.attention,
            'experiment': self.experiment.name,  # enum to string
            'rl_framework': self.rl_framework,
            'lr': self.lr,
            'gamma': self.gamma,
            'max_epsilon': self.max_epsilon,
            'min_epsilon': self.min_epsilon,
            'decay': self.decay,
            'agents': self.agents,
            'tau': self.tau,
            'max_steps': self.max_steps,
            'evaluation_steps': self.evaluation_steps,
            'episodes': self.episodes,
            'image_interval': self.image_interval,
            'evaluation_interval': self.evaluation_interval,
            'n_sample_points': self.n_sample_points
        }

def load_config_from_yaml(path: str) -> ExperimentConfig:
    with open(path, 'r') as file:
        config_data = yaml.safe_load(file)

    # Convert 'experiment' string to Experiment enum if present
    if 'experiment' in config_data:
        config_data['experiment'] = Experiment[config_data['experiment']]

    return ExperimentConfig(**config_data)

def parse_args():
    parser = argparse.ArgumentParser(description="DQN Agent Main Script")

    # Task to perform (train / eval / test)
    parser.add_argument(
        "-t", "--task",
        choices=["train", "eval", "test"],
        required=True,
        help="Specify the task to run: 'train', 'eval', or 'test'."
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Name of the model to load/save from."
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode."
    )

    # Optional overrides for YAML config (if needed)
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)."
    )

    return parser.parse_args()