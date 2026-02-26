# configs/validation.py
import yaml
from pydantic import BaseModel

# define the schemas for our training and model configurations
class TrainingConfig(BaseModel):
    train_split: float
    batch_size: int
    learning_rate: float
    epochs: int

class ModelConfig(BaseModel):
    input_size: int
    num_classes: int

# To simplify our definitions we can create a single class that have definitions from both TrainingConfig and ModelConfig - has its benefits but is not always necessary to do it this way
class FullConfig(BaseModel):
    training: TrainingConfig
    model: ModelConfig

# this function will load the YAML file and run a schema check on the loaded yaml
def load_config(path: str) -> FullConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return FullConfig(**config_dict) # this performs the validation
