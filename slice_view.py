import utils.visualiser as viz
from baseline.BaseDataLoader import DataLoader
import constants as c
from baseline.BaseUnet import BaseUNetTrainer
import utils.logger as logs

def main():
    dataLoader = DataLoader(c.DATASET_FOLDER)

    logger = logs.setup_logger(False)

    trainer = BaseUNetTrainer(
        image_list=['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10'],
        dataLoader=dataLoader,
        logger=logger
    )

    #trainer.create_distance_fields()
    trainer.train()
    #trainer.load_model()
    #trainer.show_DF_prediction('n11')
    #viz.create_slice_app(image_name='n2', dataLoader=dataLoader)

if __name__ == "__main__":
    main()