import argparse

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

    parser.add_argument(
        "--model_type",
        type=str,
        default="Network3D",
        help="Network Type."
    )

    return parser.parse_args()