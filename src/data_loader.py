# src/data_loader.py

torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset"""
    return X, y, label_encoder

def split_data(X, y, train_ratio=0.8):
    """Split data into train and test sets"""
    return X_train, X_test, y_train, y_test

def create_data_loaders(X_train, y_train, batch_size=32):
    """Create PyTorch data loaders"""
    return train_loader