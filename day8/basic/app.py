from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load('model.pkl')

# Define request model
class ModelInput(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "FastAPI model is running!"}

@app.post("/predict")
def predict(data: ModelInput):
    features = np.array(data.features).reshape(1, -1)  # Ensure correct shape
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}
