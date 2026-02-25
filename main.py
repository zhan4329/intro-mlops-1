import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from src.config import DATA_PATH, MODEL_PATH, PLOT_PATH, LOGS_PATH, \
    TRAIN_SPLIT, BATCH_SIZE, LEARNING_RATE, EPOCHS, INPUT_SIZE, NUM_CLASSES
from src.data_loader import load_and_preprocess_data, split_data, create_data_loaders
from src.neural_net import SimpleNN
from src.trainer import train_model
from src.visualization import plot_training_loss, plot_accuracy

# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def main():
    # Load and prepross data
    X, y, label_encoder = load_and_preprocess_data(DATA_PATH / 'data.csv')

    # Split data 80/20
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=TRAIN_SPLIT)

    # Create data loaders
    train_loader = create_data_loaders(X_train, y_train, batch_size=BATCH_SIZE)

    # Initialize model, loss function, optimizer, and set epochs
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))

    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epochs = EPOCHS

    # Train model
    train_losses, test_losses, accuracies = train_model(
        model=model, train_loader=train_loader, 
        X_test=X_test, y_test=y_test, criterion=criterion, 
        optimizer=optimizer, epochs=epochs)

    # Plot metrics
    plot_training_loss(train_losses, test_losses, 
        save_path=PLOT_PATH / 'training_loss_plot.png')
    plot_accuracy(accuracies, 
        save_path=PLOT_PATH / 'accuracy_plot.png')

    # Save some results to a file
    accuracy = accuracies[-1]
    print("Final Test Accuracy: ", round(accuracy, 4))
    with open(LOGS_PATH / 'results.log', 'w') as f:
        f.write(f"Final Test Accuracy: {accuracy}\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Model architecture: SimpleNN with {input_size} input features\n")

    # Save model weights
    torch.save(model.state_dict(), MODEL_PATH / "best_model.pth")
    print("Model saved to models/best_model.pth")

    # Save the label encoder
    import pickle
    with open(MODEL_PATH / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved to models/label_encoder.pkl")

    print("All done!") 

if __name__ == "__main__":
    main()
