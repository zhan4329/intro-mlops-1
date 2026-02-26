# src/config.py
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

# File paths
DATA_PATH = Path("/anvil/projects/tdm/data/wine").expanduser()
CONFIG_PATH = ROOT_PATH / 'configs'
MODEL_PATH = ROOT_PATH / 'models'
BASE_DIR = ROOT_PATH
# PLOT_PATH = ROOT_PATH / 'plots'
# LOGS_PATH = ROOT_PATH / 'logs'

# Model parameters
# TRAIN_SPLIT = 0.8
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
# EPOCHS = 5
# INPUT_SIZE = 11
# NUM_CLASSES = 3