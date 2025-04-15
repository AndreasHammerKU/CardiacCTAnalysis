import utils.logger as logs
from utils.parser import ExperimentConfig, load_config_from_yaml, parse_args
from bin.Environment import MedicalImageEnvironment
from bin.DataLoader import DataLoader
from bin.Trainer import Trainer
import constants as c

def train_model(config : ExperimentConfig, model_name, logger, dataLoader : DataLoader):
    # Train split
    train_env = MedicalImageEnvironment(logger=logger,
                                        task="train",
                                        dataLoader=dataLoader, 
                                        image_list=dataLoader.train, 
                                        n_sample_points=config.n_sample_points,
                                        agents=config.agents)
    
    # Evaluation split
    eval_env = MedicalImageEnvironment(logger=logger,
                                       task="eval",
                                       dataLoader=dataLoader,
                                       image_list=dataLoader.val,
                                       n_sample_points=config.n_sample_points,
                                       agents=config.agents)
    
    trainer = Trainer(train_environment=train_env,
                     eval_environment=eval_env,
                     task="train",
                     logger=logger,
                     dataLoader=dataLoader,
                     action_dim=train_env.n_actions,
                     attention=config.attention,
                     model_name=model_name,
                     model_type=config.model_type,
                     max_steps=config.max_steps,
                     episodes=config.episodes,
                     image_interval=config.image_interval,
                     experiment=config.experiment,
                     rl_framework=config.rl_framework,
                     min_epsilon=config.min_epsilon,
                     max_epsilon=config.max_epsilon,
                     decay=config.decay,
                     evaluation_interval=config.evaluation_interval,
                     evaluation_steps=config.evaluation_steps,
                     lr=config.lr,
                     gamma=config.gamma,
                     tau=config.tau)
    
    trainer.train()
    train_env.visualize_current_state()
    eval_env.visualize_current_state()

def test_model(config : ExperimentConfig, model_name, logger, dataLoader : DataLoader, external=False):
    # Evaluation split
    test_env = MedicalImageEnvironment(logger=logger,
                                       task="eval",
                                       dataLoader=dataLoader,
                                       image_list=dataLoader.test_external if external else dataLoader.test,
                                       n_sample_points=config.n_sample_points,
                                       agents=config.agents)
    
    trainer = Trainer(test_environment=test_env,
                     task="train",
                     logger=logger,
                     dataLoader=dataLoader,
                     action_dim=test_env.n_actions,
                     attention=config.attention,
                     model_name=model_name,
                     model_type=config.model_type,
                     max_steps=config.max_steps,
                     episodes=config.episodes,
                     image_interval=config.image_interval,
                     experiment=config.experiment,
                     rl_framework=config.rl_framework,
                     min_epsilon=config.min_epsilon,
                     max_epsilon=config.max_epsilon,
                     decay=config.decay,
                     evaluation_interval=config.evaluation_interval,
                     evaluation_steps=config.evaluation_steps,
                     lr=config.lr,
                     gamma=config.gamma,
                     tau=config.tau)
    trainer.test()
    test_env.visualize_current_state()

def main():
    args = parse_args()

    logger = logs.setup_logger(args.debug)
    
    dataLoader = DataLoader(c.DATASET_FOLDER, logger=logger, seed=1)
    
    config = load_config_from_yaml(args.config)

    if args.task == "train":
        train_model(config=config, model_name=args.model, dataLoader=dataLoader, logger=logger)

    if args.task == "test":
        test_model(config=config, model_name=args.model, dataLoader=dataLoader, logger=logger)

    if args.task == "test_external":
        test_model(config=config, model_name=args.model, dataLoader=dataLoader, logger=logger, external=True)

if __name__ == "__main__":
    main()
