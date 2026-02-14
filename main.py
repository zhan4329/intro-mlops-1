import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from src.data_loader import load_and_preprocess_data, split_data, create_data_loaders
from src.neural_net import SimpleNN
from src.trainer import train_model
from src.visualization import plot_training_loss, plot_accuracy

# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

"""
Using Path objects makes it really easy to modify paths when we refactor - look up pathlib docs if you want to learn more!
Normally we would put these in a gloabl file and import them but we will not do that for this project (but feel free too!)
"""
# Once you get root_directory_path set to point at the root of the working dir, you should NOT change it
# only change the relative paths like data_path, model_path etc
project_dir = "~/seminar-project/project4"
root_directory_path = Path(f"{project_dir}/intro-mlops-1/").expanduser()
data_path =  root_directory_path / "data" / "data.csv" # NOTE: make sure to update the path when you create the new directories i.e / "data" / "data.csv"

def main():
    # Load and prepross data
    X, y, label_encoder = load_and_preprocess_data(data_path)

    # Split data 80/20
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio=0.8)

    # Create data loaders
    train_loader = create_data_loaders(X_train, y_train, batch_size=32)

    # Initialize model, loss function, optimizer, and set epochs
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y))

    model = SimpleNN(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50

    # Train model
    train_losses, test_losses, accuracies = train_model(
        model=model, train_loader=train_loader, 
        X_test=X_test, y_test=y_test, criterion=criterion, 
        optimizer=optimizer, epochs=epochs)

    # Plot metrics
    plot_directory = root_directory_path / "plots" # NOTE: change to "plots" when you make new directory
    plot_training_loss(train_losses, test_losses, 
        save_path=plot_directory / 'training_loss_plot.png')
    plot_accuracy(accuracies, 
        save_path=plot_directory / 'accuracy_plot.png')

    # Save some results to a file
    log_dir = root_directory_path / "logs"
    accuracy = accuracies[-1]
    print("Final Test Accuracy: ", round(accuracy, 4))
    with open(log_dir / 'results.log', 'w') as f:
        f.write(f"Final Test Accuracy: {accuracy}\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Model architecture: SimpleNN with {input_size} input features\n")

    print("All done!") 

if __name__ == "__main__":
    main()
