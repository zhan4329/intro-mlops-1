from fastapi import FastAPI
import torch
import pickle

from src.config import INPUT_SIZE, NUM_CLASSES, MODEL_PATH
from src.neural_net import SimpleNN

app = FastAPI()

# loading the model up
# input_size = 4 # Iris dataset has 4 features
# num_classes = 3
model = SimpleNN(INPUT_SIZE, NUM_CLASSES) # recreating the model architecture!
model.load_state_dict(torch.load(MODEL_PATH / "best_model.pth"))
model.eval()

with open(MODEL_PATH / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.post("/predict")
def predict(data: list[float]):
    features = torch.FloatTensor(data).unsqueeze(0)

    with torch.no_grad():
        outputs = model(features)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = label_encoder.inverse_transform(predicted_idx.numpy())[0] # decodes the encoded label

    return {"prediction": predicted_label}