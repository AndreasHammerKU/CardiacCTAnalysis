import utils.logger as logs
from utils.parser import ExperimentConfig, load_config_from_yaml, parse_args
from baseline.BaseEnvironment import MedicalImageEnvironment
from baseline.BaseDataLoader import DataLoader
from baseline.BaseAgent import DQNAgent
import constants as c

def train_model(config : ExperimentConfig, model_name, logger, dataLoader):
    # Train split
    train_env = MedicalImageEnvironment(logger=logger,
                                        task="train",
                                        dataLoader=dataLoader, 
                                        image_list=['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18', 'n19', 'n20', 'n21', 'n22', 'n23', 'n24', 'n25', 'n26', 'n27', 'n28', 'n29', 'n30'], 
                                        n_sample_points=config.n_sample_points,
                                        agents=config.agents,
                                        use_unet=config.use_unet,
                                        unet_init_features=config.unet_init_features)
    
    # Evaluation split
    eval_env = MedicalImageEnvironment(logger=logger,
                                       task="eval",
                                       dataLoader=dataLoader,
                                       image_list=['n31', 'n32', 'n33', 'n34', 'n35', 'n36', 'n37', 'n38', 'n39', 'n40'],
                                       n_sample_points=config.n_sample_points,
                                       agents=config.agents,
                                       use_unet=config.use_unet,
                                       unet_init_features=config.unet_init_features)
    
    agent = DQNAgent(train_environment=train_env,
                     eval_environment=eval_env,
                     task="train",
                     logger=logger,
                     dataLoader=dataLoader,
                     state_dim=train_env.state_size,
                     action_dim=train_env.n_actions,
                     attention=config.attention,
                     model_name=model_name,
                     model_type=config.model_type,
                     max_steps=config.max_steps,
                     episodes=config.episodes,
                     image_interval=config.image_interval,
                     experiment=config.experiment,
                     min_epsilon=config.min_epsilon,
                     max_epsilon=config.max_epsilon,
                     decay=config.decay,
                     evaluation_interval=config.evaluation_interval,
                     evaluation_steps=config.evaluation_steps,
                     lr=config.lr,
                     gamma=config.gamma,
                     tau=config.tau,
                     use_unet=config.use_unet)
    
    agent.train_dqn()
    train_env.visualize_current_state()
    eval_env.visualize_current_state()

def test_model(config : ExperimentConfig, model_name, logger, dataLoader):
    # Evaluation split
    test_env = MedicalImageEnvironment(logger=logger,
                                       task="eval",
                                       dataLoader=dataLoader,
                                       image_list=['n31', 'n32', 'n33', 'n34', 'n35', 'n36', 'n37', 'n38', 'n39', 'n40'],
                                       n_sample_points=config.n_sample_points,
                                       agents=config.agents,
                                       use_unet=config.use_unet,
                                       unet_init_features=config.unet_init_features)
    
    agent = DQNAgent(test_environment=test_env,
                     task="train",
                     logger=logger,
                     dataLoader=dataLoader,
                     state_dim=test_env.state_size,
                     action_dim=test_env.n_actions,
                     attention=config.attention,
                     model_name=model_name,
                     model_type=config.model_type,
                     max_steps=config.max_steps,
                     episodes=config.episodes,
                     image_interval=config.image_interval,
                     experiment=config.experiment,
                     min_epsilon=config.min_epsilon,
                     max_epsilon=config.max_epsilon,
                     decay=config.decay,
                     evaluation_interval=config.evaluation_interval,
                     evaluation_steps=config.evaluation_steps,
                     lr=config.lr,
                     gamma=config.gamma,
                     tau=config.tau,
                     use_unet=config.use_unet)
    agent.test_dqn()
    test_env.visualize_current_state()

def main():
    args = parse_args()

    logger = logs.setup_logger(args.debug)
    
    dataLoader = DataLoader(c.DATASET_FOLDER)
    
    config = load_config_from_yaml(args.config)

    if args.task == "train":
        train_model(config=config, model_name=args.model, dataLoader=dataLoader, logger=logger)

    if args.task == "test":
        test_model(config=config, model_name=args.model, dataLoader=dataLoader, logger=logger)

if __name__ == "__main__":
    main()
