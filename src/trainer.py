# src/trainer.py
torch.manual_seed(42)

def train_one_epoch(model, train_loader, optimizer, criterion):
    """Helper function to complete one training epoch and return average train loss"""

    return avg_train_loss

def evaluate_model(model, criterion, X_test, y_test):
    """Helper function to evaluate the model after each training epoch"""

    return accuracy, test_loss


def train_model(model, train_loader, X_test, y_test, criterion, optimizer, epochs=50):
    """Higher level training driver function, returns metric arrays"""

    return train_losses, test_losses, accuracies