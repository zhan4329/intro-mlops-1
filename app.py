# app.py
import torch
import pickle
import mlflow

from intro_mlops_2.configs.config import MODEL_PATH
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
from intro_mlops_2.configs.api_validation import WineQualityRequest, WineQualityResponse

def get_latest_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name("WineQualityClassifier")
    latest_run = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)[0]
    model_uri = f"runs:/{latest_run.info.run_id}/model"
    return mlflow.pytorch.load_model(model_uri)

app = FastAPI()
model = get_latest_model()
model.eval()

with open(MODEL_PATH / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.post("/predict", response_model=WineQualityResponse)
async def predict(input_data: WineQualityRequest):
    features = torch.FloatTensor([
        input_data.fixed_acidity,
        input_data.volatile_acidity,
        input_data.citric_acid,
        input_data.residual_sugar,
        input_data.chlorides,
        input_data.free_sulfur_dioxide,
        input_data.total_sulfur_dioxide,
        input_data.density,
        input_data.pH,
        input_data.sulphates,
        input_data.alcohol,
        # ... other feature fields in the same order as your training data
    ]).unsqueeze(0)

    with torch.no_grad():
        logits = model(features)
    pred_idx = logits.argmax(dim=1).item()
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    return {"quality": ["Bad", "Mid", "Good"][pred_idx]}