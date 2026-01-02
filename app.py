from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # beginner-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_model = joblib.load("../ml/models/logistic_model.pkl")
dt_model = joblib.load("../ml/models/decision_tree_model.pkl")
scaler = joblib.load("../ml/models/scaler.pkl")

@app.post("/predict")
def predict(data: dict):
    gender = data["gender"]
    age = data["age"]
    income = data["income"]

    features = np.array([[gender, age, income]])
    features = scaler.transform(features)

    log_pred = log_model.predict(features)[0]
    dt_pred = dt_model.predict(features)[0]

    return {
        "logistic_regression": int(log_pred),
        "decision_tree": int(dt_pred)
    }
