import logging
import pandas as pd
from datetime import datetime
import os

class MedicalLogger:
    def __init__(self, debug=False):
        self.logger = logging.getLogger("Logger")

        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler("log.txt")
        console_handler = logging.StreamHandler()

        # Add handlers to logger
        if not self.logger.hasHandlers():  # Avoid duplicate handlers in case of re-runs
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
    
    def info(self, string):
        self.logger.info(string)
    
    def debug(self, string):
        self.logger.debug(string)


    # === Data Management ===
    def create_dataframes(self):
        self.debug("Creating new empty DataFrames.")
        self.train_df = pd.DataFrame(columns=["episode", "total_reward", "end_avg_dist", "avg_err_in_mm", "worst_err_in_mm", "avg_closest_point", "avg_furthest_point"])
        self.val_df = pd.DataFrame(columns=["train_episode", "episode", "total_reward", "end_avg_dist", "avg_err_in_mm", "worst_err_in_mm", "avg_closest_point", "avg_furthest_point", "naive_distance_mm", "CPD_distance_mm"])
    def insert_train_row(self, episode, total_reward, end_avg_dist, avg_err_in_mm, worst_err_in_mm, avg_closest_point, avg_furthest_point):
        self.debug(f"Inserting row into train_df: episode={episode}")
        self.train_df.loc[len(self.train_df)] = [episode, total_reward, end_avg_dist, avg_err_in_mm, worst_err_in_mm, avg_closest_point, avg_furthest_point]

    def insert_val_row(self, train_episode, episode, total_reward, end_avg_dist, avg_err_in_mm, worst_err_in_mm, avg_closest_point, avg_furthest_point, naive_distance_mm, CPD_distance_mm):
        self.debug(f"Inserting row into val_df: episode={episode}")
        self.val_df.loc[len(self.val_df)] = [train_episode, episode, total_reward, end_avg_dist, avg_err_in_mm, worst_err_in_mm, avg_closest_point, avg_furthest_point, naive_distance_mm, CPD_distance_mm]

    def clear_dataframes(self):
        self.debug("Clearing existing DataFrames.")
        self.create_dataframes()

    def generate_run_id(self, config_dict):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_id = f"{config_dict['rl_framework']}-{config_dict['model_type']}-{config_dict['experiment']}-{timestamp}"
        filename = run_id + ".h5"
        self.debug(f"Generated run_id: {run_id}, filename: {filename}")
        return run_id, filename

    # --- Save & Load to/from HDF5 ---
    def save_to_hdf5(self, config_obj, directory='.'):
        if not hasattr(config_obj, 'to_dict'):
            raise ValueError("Config object must have a to_dict() method.")
        config_dict = config_obj.to_dict()
        run_id, filename = self.generate_run_id(config_dict)
        filename = os.path.join(directory, filename)
        self.debug(f"Saving training and validation data to {filename}")
        with pd.HDFStore(filename) as store:
            store["train_epochs"] = self.train_df
            store["val_epochs"] = self.val_df
            store.get_storer("train_epochs").attrs.config = config_dict
            store.get_storer("train_epochs").attrs.run_id = run_id

        return filename

    def load_from_hdf5(self, filename):
        self.debug(f"Loading training and validation data from {filename}")
        with pd.HDFStore(filename) as store:
            train_df = store["train_epochs"]
            val_df = store["val_epochs"]
            config = store.get_storer("train_epochs").attrs.config
            run_id = store.get_storer("train_epochs").attrs.run_id

        return train_df, val_df, config, run_id
    

def setup_logger(debug=False):
    return MedicalLogger(debug=debug)