from bin.DataLoader import DataLoader
import utils.logger as logs
import constants as c
from utils.visualiser import visualize_from_logs

logger = logs.setup_logger(False)

visualize_from_logs(logger=logger, experiment='DQN-Final', save_path=c.FIGURE_FOLDER, viz_name="DQN-Final-Final")