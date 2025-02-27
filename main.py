import utils.logger as logs
import utils.parser as parse
from baseline.BaseEnvironment import MedicalImageEnvironment
from utils.io_utils import DataLoader
from baseline.BaseAgent import DQNAgent
import constants as c

def main():
    args = parse.parse_args()

    logger = logs.setup_logger(args.debug)
    if args.dataset is None:
        dataLoader = DataLoader(c.DATASET_FOLDER)
    else:
        dataLoader = DataLoader(args.dataset)
    
    if args.task == "train":
        # Train split
        train_env = MedicalImageEnvironment(logger=logger, 
                                  dataLoader=dataLoader, 
                                  image_list=['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18', 'n19', 'n20', 'n21', 'n22', 'n23', 'n24', 'n25', 'n26', 'n27', 'n28', 'n29', 'n30'], 
                                  agents=1,
                                  n_sample_points=args.samples,
                                  preload_images=args.preload)
        
        # Evaluation split
        eval_env = MedicalImageEnvironment(logger=logger,
                                      task="eval",
                                      dataLoader=dataLoader,
                                      image_list=['n31', 'n32', 'n33', 'n34', 'n35', 'n36', 'n37', 'n38', 'n39', 'n40'],
                                      agents=1,
                                      n_sample_points=args.samples)
        
        agent = DQNAgent(train_environment=train_env,
                         eval_environment=eval_env,
                         task="train",
                         logger=logger,
                         state_dim=train_env.state_size,
                         action_dim=train_env.n_actions,
                         agents=1,
                         model_path=args.model,
                         max_steps=args.steps,
                         episodes=args.episodes,
                         image_interval=args.image_interval)
    
        agent.train_dqn()

    if args.task == "test":
        # Test split
        test_env = MedicalImageEnvironment(logger=logger,
                                      task="test",
                                      dataLoader=dataLoader,
                                      image_list=['n41', 'n42', 'n43', 'n44', 'n45', 'n46', 'n47', 'n48', 'n49', 'n50'],
                                      agents=1,
                                      n_sample_points=args.samples)
        
        agent = DQNAgent(test_environment=test_env,
                         task="eval",
                         logger=logger,
                         state_dim=test_env.state_size,
                         action_dim=test_env.n_actions,
                         agents=1,
                         model_path=args.model,
                         max_steps=args.steps,
                         episodes=args.episodes)
        
        agent.test_dqn()

if __name__ == "__main__":
    main()
