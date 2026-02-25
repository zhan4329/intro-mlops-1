# src/lightning.py

import pytorch_lightning as pl
import torch
from torch import nn

class WineQualityClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate, epochs):
        super().__init__()
        self.save_hyperparameters(ignore=["model"]) # don't serialize the model object
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss() # cross entropy loss is a common loss function for classification tasks

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