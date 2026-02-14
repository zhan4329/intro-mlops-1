# src/metrics.py

def calculate_accuracy(predicted, y_test):
    """Calculate accuracy"""
    accuracy = (predicted == y_test).sum().item() / len(y_test)    
    return accuracy