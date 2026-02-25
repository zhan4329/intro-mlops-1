import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from intro_mlops_2.src.lightning import WineQualityClassifier
from intro_mlops_2.src.data_loader import WineQualityDataModule

from src.config import DATA_PATH, MODEL_PATH, PLOT_PATH, LOGS_PATH, \
    TRAIN_SPLIT, BATCH_SIZE, LEARNING_RATE, EPOCHS, INPUT_SIZE, NUM_CLASSES
# from src.data_loader import load_and_preprocess_data, split_data, create_data_loaders
from src.neural_net import SimpleNN
# from src.trainer import train_model
from src.visualization import plot_training_loss, plot_accuracy

# set seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def main():
    # Initialize model and dataloader
    simple_nn = SimpleNN(INPUT_SIZE, NUM_CLASSES)
    model = WineQualityClassifier(simple_nn, learning_rate=LEARNING_RATE, epochs=EPOCHS)
    datamodule = WineQualityDataModule(data_path=DATA_PATH / "wine_quality_type.csv", batch_size=BATCH_SIZE, train_split=TRAIN_SPLIT)

    # Initialize trainer object and train
    trainer = pl.Trainer(max_epochs=model.hparams.epochs)
    trainer.fit(model, datamodule=datamodule)

    # Train model
    # train_losses, test_losses, accuracies = train_model(
    #     model=model, train_loader=train_loader, 
    #     X_test=X_test, y_test=y_test, criterion=criterion, 
    #     optimizer=optimizer, epochs=epochs)

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
        f.write(f"Training epochs: {model.hparams.epochs}\n")
        f.write(f"Model architecture: SimpleNN with {INPUT_SIZE} input features\n")

    # Save model weights
    torch.save(model.state_dict(), MODEL_PATH / "best_model.pth")
    print("Model saved to models/best_model.pth")

    # Save the label encoder
    import pickle
    with open(MODEL_PATH / "label_encoder.pkl", "wb") as f:
        pickle.dump(datamodule.label_encoder, f)
    print("Label encoder saved to models/label_encoder.pkl")

    print("All done!") 

if __name__ == "__main__":
    main()
