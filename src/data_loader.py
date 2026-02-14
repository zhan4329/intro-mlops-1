# src/data_loader.py
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset"""
    print("Loading dataset...")
    data = pd.read_csv(data_path)
    print(f"Dataset shape: {data.shape}")

    print("Preprocessing data...")
    # Remove any missing values
    data = data.dropna()

    # Splitting data into features and target
    feature_cols = data.columns[:-1].tolist()
    target_col = data.columns[-1]

    X = data[feature_cols].to_numpy()
    y = data[target_col].to_numpy()

    # Predicting string labels so encode them
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X, y, label_encoder

def split_data(X, y, train_ratio=0.8):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=42, shuffle=True
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Convert to PyTorch tensors (required)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def create_data_loaders(X_train, y_train, batch_size=32):
    """Create PyTorch data loaders"""
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader