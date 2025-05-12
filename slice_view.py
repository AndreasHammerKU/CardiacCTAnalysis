import utils.visualiser as viz
from bin.DataLoader import DataLoader
import constants as c
from utils.logger import setup_logger

def main():
    logger = setup_logger(debug=True)
    dataLoader = DataLoader(c.DATASET_FOLDER, logger=logger)

    viz.create_slice_app(image_name='n2', dataLoader=dataLoader)

if __name__ == "__main__":
    main()