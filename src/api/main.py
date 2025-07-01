from fastapi import FastAPI
import mlflow.sklearn
import numpy as np
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

app = FastAPI()

# Load model from MLflow
model_uri = "models:/best_model/Production"  # Change name/version as needed
model = mlflow.sklearn.load_model(model_uri)

@app.get("/")
def root():
    return {"message": "Credit Risk API is up."}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: CustomerFeatures):
    X = np.array(data.features).reshape(1, -1)
    probability = model.predict_proba(X)[0][1]
    return {"risk_probability": round(probability, 4)}
