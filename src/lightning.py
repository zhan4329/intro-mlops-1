# src/lightning.py

import pytorch_lightning as pl
import torch

from torch import nn
from intro_mlops_2.configs.validation import load_config

class WineQualityClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate, epochs):
        super().__init__()
        self.save_hyperparameters(ignore=["model"]) # don't serialize the model object
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss() # cross entropy loss is a common loss function for classification tasks

    @classmethod
    def from_config_path(cls, model_cls: nn.Module, config_path: str):
        config = load_config(config_path)
        model = model_cls( # instantiate the model with the input size and number of classes from the config
            input_size=config.model.input_size,
            num_classes=config.model.num_classes,
        )
        return cls( # return the class with the model, learning rate, and epochs from the config
             model=model,
             learning_rate=config.training.learning_rate,
             epochs=config.training.epochs,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch                      # splits out data (`x`) and labels (`y`)
        logits = self(x)                  # passes data into models forward method
        loss = self.loss_fn(logits, y)    # calcs loss based on logits (output of model) and ground truth
        self.log("train_loss", loss)      # this is a special method that logs the loss at each step
        return loss

    def configure_optimizers(self):
        # uses self.hparams
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def validation_step(self, batch, batch_idx):
        x, y = batch                      # splits out data (`x`) and labels (`y`)
        logits = self(x)                  # passes data into models forward method
        loss = self.loss_fn(logits, y)    # calcs loss based on logits (output of model) and ground truth
        preds = logits.argmax(dim=1)      # predicted class per sample
        acc = (preds == y).float().mean() # fraction correct
        self.log("val_loss", loss)
        self.log("val_accuracy", acc)     # so you can see accuracy in results and in callbacks
        return loss