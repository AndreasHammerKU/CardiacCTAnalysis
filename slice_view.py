import utils.visualiser as viz
from utils.io_utils import DataLoader
import constants as c
from baseline.BaseEnvironment import MedicalImageEnvironment
import utils.logger as logs
import utils.parser as parse
def main():
    dataLoader = DataLoader(c.DATASET_FOLDER)

    logger = logs.setup_logger(True)

    env = MedicalImageEnvironment(
        task="train",
        image_list=['n1', 'n2'],
        dataLoader=dataLoader,
        logger=logger
    )
    env.get_next_image()
    env.reset()
    

    #env.visualize_current_state()
    env.get_distance_field(granularity=10)


    #viz.create_slice_app('n2', dataLoader=dataLoader)


if __name__ == "__main__":
    main()