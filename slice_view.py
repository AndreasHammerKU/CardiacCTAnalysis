import utils.visualiser as viz
from utils.io_utils import DataLoader
import constants as c

def main():
    dataLoader = DataLoader(c.DATASET_FOLDER)

    viz.create_slice_app('n2', dataLoader=dataLoader)


if __name__ == "__main__":
    main()