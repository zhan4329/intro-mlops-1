# main.py
import mlflow
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from intro_mlops_2.configs.config import *
from intro_mlops_2.src.data_loader import WineQualityDataModule
from intro_mlops_2.src.lightning import WineQualityClassifier
from intro_mlops_2.src.neural_net import SimpleNN


# Create model and datamodule from config
model = WineQualityClassifier.from_config_path(SimpleNN, CONFIG_PATH / "model.yaml")
datamodule = WineQualityDataModule.from_config_path(CONFIG_PATH / "model.yaml", DATA_PATH / "wine_quality_type.csv")


mlflow.set_experiment("WineQualityClassifier") # this is the name of the experiment that will be created in the MLflow UI

with mlflow.start_run():

    # Log hyperparameters (from our config / Lightning module)
    mlflow.log_params({
        "epochs": model.hparams.epochs,
        "learning_rate": model.hparams.learning_rate,
        "model_class": model.model.__class__.__name__,
    })

    # callbacks are instantiated here
    early_stopping = EarlyStopping(
        monitor="val_loss",    # which metric to watch
        patience=10,           # how many epochs to wait before stopping
        mode="min"             # stop when val_loss stops decreasing
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",                              # which metric to track (val_loss is being logged in the validation_step method)
        dirpath=ROOT_PATH / "models",                     # where to save checkpoints
        filename="best-{epoch:02d}-{val_loss:.2f}",      # filename format
        save_top_k=1,                                    # only save the best 1 model
        mode="min"                                       # save when val_loss is minimum
    )
    # the callbacks are then attached to the trainer object
    # Create a logger so metrics are also saved to a CSV file in a custom directory
    # csv_logger = CSVLogger(save_dir=str(BASE_DIR / "logs"), name="wine_quality")
    # Add callbacks and logger to trainer
    trainer = pl.Trainer(
        max_epochs=model.hparams.epochs,
        callbacks=[early_stopping, checkpoint_callback],
        # logger=csv_logger,
        # enable_progress_bar=False # this causes issues in the notebook, so we disable it, if you are running this in the command line i would recommend leaving it on
    )
    trainer.fit(model, datamodule=datamodule)

    # Log final validation metrics
    val_results = trainer.validate(model, datamodule=datamodule)
    if val_results:
        for k, v in val_results[0].items():
            mlflow.log_metric(k, v)

    # Save the Lightning model with MLflow (full model; no need to reconstruct later)
    # Get a sample batch to use as input_example for model signature
    sample_batch = next(iter(datamodule.val_dataloader()))
    sample_input = sample_batch[0][:1]  # first sample from validation batch
    mlflow.pytorch.log_model(model, name="model", input_example=sample_input.numpy())

# Now we are training with callbacks and logging enabled (disabled progress bar since in notebook so added some prints to confirm training is happening)
# def print_box(text, width=80):
#     print("=" * width)
#     print("|" + text.center(width - 2) + "|")
#     print("=" * width)

# print_box("Training started...")
# trainer.fit(model, datamodule=datamodule)
# print_box("Training completed!")

# Validate the model (output includes val_loss and val_accuracy)
# val_results = trainer.validate(model, datamodule=datamodule)

# remove all the code to graph and save model weights since we are not using them anymore.
