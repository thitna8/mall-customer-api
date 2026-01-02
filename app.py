from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="Mall Customer Prediction API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mall-customer-1fkqgonjm-thitna8s-projects.vercel.app"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models from models folder (relative path)
log_model = joblib.load("models/logistic_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.post("/predict")
def predict(data: dict):
    """
    Expects JSON with keys: gender, age, income
    Returns predictions from Logistic Regression and Decision Tree
    """
    gender = data["gender"]
    age = data["age"]
    income = data["income"]

    # Prepare features
    features = np.array([[gender, age, income]])
    features = scaler.transform(features)

    # Predict
    log_pred = log_model.predict(features)[0]
    dt_pred = dt_model.predict(features)[0]

    return {
        "logistic_regression": int(log_pred),
        "decision_tree": int(dt_pred)
    }
