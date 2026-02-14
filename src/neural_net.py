import torch
import torch.nn as nn

# Simple NN declaration
class SimpleNN(nn.Module):
  def __init__(self, input_size, num_classes=3):
    super(SimpleNN, self).__init__()
    self.layer1 = nn.Linear(input_size, 64)
    self.layer2 = nn.Linear(64, 32)
    self.layer3 = nn.Linear(32, num_classes)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    
  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.dropout(x)
    x = self.relu(self.layer2(x))
    x = self.dropout(x)
    x = self.layer3(x)
    return x
