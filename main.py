import pytorch_lightning as pl

from intro_mlops_2.src.lightning import WineQualityClassifier
from intro_mlops_2.src.data_loader import WineQualityDataModule
from intro_mlops_2.src.config import DATA_PATH, MODEL_PATH, PLOT_PATH, LOGS_PATH, \
    TRAIN_SPLIT, BATCH_SIZE, LEARNING_RATE, EPOCHS, INPUT_SIZE, NUM_CLASSES
from intro_mlops_2.src.neural_net import SimpleNN


def main():
    # Initialize model and dataloader
    simple_nn = SimpleNN(INPUT_SIZE, NUM_CLASSES)
    model = WineQualityClassifier(simple_nn, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    datamodule = WineQualityDataModule(data_path=DATA_PATH / "wine_quality_type.csv", batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT)

    # Initialize trainer object and train
    trainer = pl.Trainer(max_epochs=model.hparams.epochs)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
