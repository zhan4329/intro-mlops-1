# main.py
import pytorch_lightning as pl

from intro_mlops_2.configs.config import *
from intro_mlops_2.src.data_loader import WineQualityDataModule
from intro_mlops_2.src.lightning import WineQualityClassifier
from intro_mlops_2.src.neural_net import SimpleNN

# Create model and datamodule from config
model = WineQualityClassifier.from_config_path(SimpleNN, CONFIG_PATH / "model.yaml")
datamodule = WineQualityDataModule.from_config_path(CONFIG_PATH / "model.yaml", DATA_PATH / "wine_quality_type.csv")

# Initialize trainer and train
trainer = pl.Trainer(max_epochs=model.hparams.epochs) # notice the hparams.epochs is being used to set the max number of epochs based on the models config
trainer.fit(model, datamodule=datamodule)

# Validate the model (output includes val_loss and val_accuracy)
val_results = trainer.validate(model, datamodule=datamodule)
print("Validation results:", val_results)

# remove all the code to graph and save model weights since we are not using them anymore.
