import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path

from src.data_loader import load_and_preprocess_data, split_data, create_data_loaders
from src.neural_net import SimpleNN
from src.trainer import train_model


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

plot_directory = root_directory_path / "" # NOTE: change to "plots" when you make new directory

# Save Training/Test Loss Plot
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Time')
plt.legend()
plt.savefig(plot_directory / 'training_loss_plot.png')
plt.close() 

# Save Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(accuracies, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Over Time')
plt.legend()
plt.savefig(plot_directory / 'accuracy_plot.png')
plt.close() 

# Save some results to a file
log_dir = root_directory_path / ""
accuracy = accuracies[-1]
print("Final Test Accuracy: ", round(accuracy, 4))
with open(log_dir / 'results.log', 'w') as f:
    f.write(f"Final Test Accuracy: {accuracy}\n")
    f.write(f"Training epochs: {epochs}\n")
    f.write(f"Model architecture: SimpleNN with {input_size} input features\n")

print("All done!") 

