# src/visualization.py
import matplotlib.pyplot as plt

def plot_training_loss(train_losses, test_losses, save_path):
    """Plot training and test losses (save_path is the full path for the output image file)"""
    # Save Training/Test Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.savefig(save_path)
    plt.close() 
    
def plot_accuracy(accuracies, save_path):
    """Plot accuracy over time (save_path is the full path for the output image file)"""
    # Save Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(accuracies, label='Test Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Over Time')
    plt.legend()
    plt.savefig(save_path)
    plt.close() 