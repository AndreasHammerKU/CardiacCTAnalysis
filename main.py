from utils.logger import setup_logger, MedicalLogger
from utils.parser import ExperimentConfig, load_config_from_yaml, parse_args
from bin.Environment import MedicalImageEnvironment, rearrange_points
from bin.DataLoader import DataLoader
from bin.Trainer import Trainer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants as c
import os

def train_model(config : ExperimentConfig, logger : MedicalLogger, dataLoader : DataLoader):
    logger.create_dataframes()
    # Train split
    train_env = MedicalImageEnvironment(logger=logger,
                                        task="train",
                                        dataLoader=dataLoader, 
                                        image_list=dataLoader.train, 
                                        n_sample_points=config.n_sample_points,
                                        agents=config.agents,
                                        add_noise=config.add_noise)
    
    # Evaluation split
    eval_env = MedicalImageEnvironment(logger=logger,
                                       task="eval",
                                       dataLoader=dataLoader,
                                       image_list=dataLoader.val,
                                       n_sample_points=config.n_sample_points,
                                       agents=config.agents,
                                       train_images=dataLoader.train,
                                       add_noise=config.add_noise)
    
    trainer = Trainer(train_environment=train_env,
                     eval_environment=eval_env,
                     task="train",
                     logger=logger,
                     dataLoader=dataLoader,
                     action_dim=train_env.n_actions,
                     attention=config.attention,
                     model_name=f"{config.rl_framework}-{config.model_type}-{config.experiment.name}",
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
    logger.save_to_hdf5(config_obj=config, directory="logs")

def test_model(config : ExperimentConfig, logger : MedicalLogger, dataLoader : DataLoader, external=False):
    logger.create_dataframes()
    # Evaluation split
    test_env = MedicalImageEnvironment(logger=logger,
                                       task="eval",
                                       dataLoader=dataLoader,
                                       image_list=dataLoader.test_external if external else dataLoader.test[1:],
                                       n_sample_points=config.n_sample_points,
                                       agents=config.agents,
                                       train_images=dataLoader.train,
                                       add_noise=config.add_noise)

    trainer = Trainer(test_environment=test_env,
                     task="test",
                     logger=logger,
                     dataLoader=dataLoader,
                     action_dim=test_env.n_actions,
                     attention=config.attention,
                     model_name=f"{config.rl_framework}-{config.model_type}-{config.experiment.name}-best-model",
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
    logger.save_to_hdf5(config_obj=config, directory="logs")

def debug_model(config : ExperimentConfig, logger : MedicalLogger, dataLoader : DataLoader):
    # Runs auxilary functions and debug statistics
    logger.create_dataframes()

    train_env = MedicalImageEnvironment(logger=logger,
                                        task="train",
                                        dataLoader=dataLoader, 
                                        image_list=dataLoader.train, 
                                        n_sample_points=config.n_sample_points,
                                        agents=config.agents,
                                        trim_image=False)
    
    full_image_set = dataLoader.train + dataLoader.val + dataLoader.test[1:]
    # Evaluation split
    eval_env = MedicalImageEnvironment(logger=logger,
                                       task="eval",
                                       dataLoader=dataLoader,
                                       image_list=full_image_set,
                                       n_sample_points=config.n_sample_points,
                                       agents=config.agents,
                                       train_images=dataLoader.train, 
                                       trim_image=False)
    trainer = Trainer(train_environment=train_env,
                     eval_environment=eval_env,
                     task="train",
                     logger=logger,
                     dataLoader=dataLoader,
                     action_dim=train_env.n_actions,
                     attention=config.attention,
                     model_name=f"{config.rl_framework}-{config.model_type}-{config.experiment.name}",
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
    
    train_env.get_next_image()
    state = train_env.reset()
    #train_env.visualize_current_state()
    pairwise_matrixes = []
    for i in range(len(full_image_set)):
        print("Image: ", i+1)
        eval_env.get_next_image()
        state = eval_env.reset()
        #eval_env.visualize_current_state()
        true, pred = eval_env.get_aortic_valve_metrics()
        #eval_env.geometry.plot(plot_geometric_heights=True, plot_basal_ring=True, plot_bezier_curves=True, plot_label_points=False)
        pairwise_matrixes.append(eval_env.pairwise_distances)

    pairwise_matrixes = np.array(pairwise_matrixes)
    avg_distances = np.mean(pairwise_matrixes, axis=0)
    var_distances = np.var(pairwise_matrixes, axis=0)

    order = [0, 3, 1, 4, 2, 5]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Average Distance Heatmap
    sns.heatmap(avg_distances[order, :][:, order], ax=axes[0], cmap='Blues', annot=True, fmt=".2f")
    axes[0].set_title("Average Pairwise Distances (mm)")
    axes[0].set_xlabel("Points")
    axes[0].set_ylabel("Points")

    # Variance Heatmap
    sns.heatmap(var_distances[order, :][:, order], ax=axes[1], cmap='Reds', annot=True, fmt=".2f")
    
    axes[1].set_title("Variance of Pairwise Distances (mm)")
    axes[1].set_xlabel("Points")
    axes[1].set_ylabel("Points")

    plt.tight_layout()
    plt.show()
    

def main():
    args = parse_args()

    logger = setup_logger(args.debug)
    
    dataLoader = DataLoader(c.DATASET_FOLDER, model_dir=os.path.join(c.LOGS_FOLDER, c.MODEL_FOLDER), logger=logger, seed=1)
    dataLoader.test_external.remove('HOM_M53_H183_W86_YA')
    dataLoader.test_external.remove('HOM_M56_H179_W75_YA')
    dataLoader.test_external.remove('HOM_M59_H184_W97_YA')
    dataLoader.test_external.remove('HOM_M62_H178_W88_YA')
    dataLoader.test_external.remove('HOM_M70_H183_W108_YA')
    
    config = load_config_from_yaml(args.config)

    if args.task == "train":
        train_model(config=config, dataLoader=dataLoader, logger=logger)

    if args.task == "test":
        test_model(config=config, dataLoader=dataLoader, logger=logger)

    if args.task == "test_external":
        test_model(config=config, dataLoader=dataLoader, logger=logger, external=True)

    if args.task == "debug":
        debug_model(config=config, dataLoader=dataLoader, logger=logger)

if __name__ == "__main__":
    main()
