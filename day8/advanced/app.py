import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.3):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, 32)
        self.layer6 = nn.Linear(32, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout with given probability

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)  # Dropout after activation
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.relu(self.layer5(x))
        x = self.layer6(x)  # No dropout on output layer
        return x

# Load the model architecture
model = FeedForwardNN(input_dim=4, output_dim=3)
model.load_state_dict(torch.load('model.pth')) 
model.eval()

# Load the scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# FastAPI setup
app = FastAPI()

class InputData(BaseModel):
    fw: float
    inl: float
    pl: float
    x: float

class OutputData(BaseModel):
    freq: float
    s11: float
    gain: float

@app.post("/predict", response_model=OutputData)
def predict(input_data: InputData):
    input_array = np.array([[input_data.fw, input_data.inl, input_data.pl, input_data.x]])
    input_scaled = scaler_X.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction_scaled = model(input_tensor).numpy()

    prediction = scaler_y.inverse_transform(prediction_scaled)
    return OutputData(freq=prediction[0][0], s11=prediction[0][1], gain=prediction[0][2])
