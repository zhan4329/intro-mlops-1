# src/data_loader.py - project 5 version

import pytorch_lightning as pl
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

class WineQualityDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, train_split=0.8): # you can add more parameters here if you want
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_split = train_split

    def setup(self, stage=None):
        # Load and preprocess data - same logic from load_and_preprocess_data()
        data = pd.read_csv(self.data_path).dropna()

        # Create quality bins
        def bin_quality(quality):
            if quality <= 4:
                return 0  # Bad
            elif quality <= 7:
                return 1  # Mid
            else:
                return 2  # Good

        # Apply binning
        data['quality_binned'] = data['quality'].apply(bin_quality)

        # Features and target
        X = data.drop(['quality', 'quality_binned', 'type'], axis=1).values # .values converts the dataframe to a numpy array
        y = data['quality_binned'].values

        # Normalize features
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        # Split data - same logic from split_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.train_split, random_state=42, shuffle=True
        )

        # Convert to tensors - saving these in self so we can use them in our dataloaders
        self.train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        self.val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    def train_dataloader(self):
        """Return the training data loader"""
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        """Return the validation data loader"""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True
        )
