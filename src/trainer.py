# src/trainer.py
import torch

from .metrics import calculate_accuracy

torch.manual_seed(42)

def train_one_epoch(model, train_loader, optimizer, criterion):
    """Helper function to complete one training epoch and return average train loss"""
    # Train one epoch - Start
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    # Train one epoch - End
    return avg_train_loss

def evaluate_model(model, criterion, X_test, y_test):
    """Helper function to evaluate the model after each training epoch"""
    # Evaluate on test set - Start
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        _, predicted = torch.max(test_outputs, 1)
        # Calculate Accuracy
        accuracy = calculate_accuracy(predicted, y_test)
    # Evaluate on test set - End
    return accuracy, test_loss

def train_model(model, train_loader, X_test, y_test, criterion, optimizer, epochs=50):
    """Higher level training driver function, returns metric arrays"""

    # Train Model loop! - Start
    print("Starting training...")
    train_losses = [] # these are for metric tracking, these will be returned from train_model()
    test_losses = []
    accuracies = []

    for epoch in range(epochs):
        # Train one epoch
        avg_train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        accuracy, test_loss = evaluate_model(
            model, criterion, X_test, y_test)
        test_losses.append(test_loss.item())
        accuracies.append(accuracy)
    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss.item():.4f}')
        # Train Model loop! - End 

    print("Training completed!")

    return train_losses, test_losses, accuracies