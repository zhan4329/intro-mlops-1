# src/config.py
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

# File paths
DATA_PATH = ROOT_PATH / 'data'
MODEL_PATH = ROOT_PATH / 'models'
PLOT_PATH = ROOT_PATH / 'plots'
LOGS_PATH = ROOT_PATH / 'logs'

# Model parameters
TRAIN_SPLIT = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
INPUT_SIZE = 4
NUM_CLASSES = 3