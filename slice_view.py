import utils.visualiser as viz
from baseline.BaseDataLoader import DataLoader
import constants as c
from baseline.BaseUnet import BaseUNetTrainer
import utils.logger as logs

def main():
    dataLoader = DataLoader(c.DATASET_FOLDER)

    logger = logs.setup_logger(False)

    trainer = BaseUNetTrainer(
        image_list=['n1', 'n2'],
        dataLoader=dataLoader,
        logger=logger
    )

    #trainer.create_distance_fields()
    trainer.train()
    #viz.create_slice_app(image_name='n2', dataLoader=dataLoader)

if __name__ == "__main__":
    main()