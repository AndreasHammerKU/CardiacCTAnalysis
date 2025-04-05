import utils.visualiser as viz
from baseline.BaseDataLoader import DataLoader
import constants as c
from baseline.BaseUnet import BaseUNetTrainer
import utils.logger as logs

def main():
    dataLoader = DataLoader(c.DATASET_FOLDER)

    logger = logs.setup_logger(False)

    trainer = BaseUNetTrainer(
        image_list=['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 'n18', 'n19', 'n20', 'n21', 'n22', 'n23', 'n24', 'n25', 'n26', 'n27', 'n28', 'n29', 'n30', 'n31', 'n32', 'n33', 'n34', 'n35', 'n36', 'n37', 'n38', 'n39', 'n40', 'n41', 'n42', 'n43', 'n44', 'n45', 'n46', 'n47', 'n48', 'n49', 'n50'],
        dataLoader=dataLoader,
        logger=logger,
        init_features=16
    )

    #trainer.create_distance_fields(max_distance=5, granularity=50)
    #trainer.show_DF_from_file(image_name='n2', axis=2, slice_index=100)
    trainer.train(n_epochs=10)
    #trainer.load_model()
    #trainer.show_DF_prediction('n2', axis=2, slice_index=100)
    #viz.create_slice_app(image_name='n2', dataLoader=dataLoader)

if __name__ == "__main__":
    main()