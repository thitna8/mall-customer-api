from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="Mall Customer Prediction API")

# Allow frontend domains (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
       "https://thitna8.github.io/mall-customer-frontend",
        "https://mall-customer-final.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models from models folder
log_model = joblib.load("models/logistic_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.get("/")
def root():
    return {"message": "Mall Customer Prediction API is live!"}

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
