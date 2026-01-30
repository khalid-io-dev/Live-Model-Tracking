
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
import os

app = FastAPI(title="Diabetes Risk Prediction API")

# Initialize Prometheus Instrumentator
Instrumentator().instrument(app).expose(app)

# Load Artifacts
ARTIFACTS = {}

def load_artifacts():
    global ARTIFACTS
    try:
        ARTIFACTS["model"] = joblib.load("final_model.pkl")
        ARTIFACTS["imputer"] = joblib.load("imputer.pkl")
        ARTIFACTS["scaler"] = joblib.load("scaler.pkl")
        ARTIFACTS["feature_names"] = joblib.load("feature_names.pkl")
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        ARTIFACTS = {}

@app.on_event("startup")
async def startup_event():
    load_artifacts()

# Pydantic Model for Input
class PatientData(BaseModel):
    Glucose: float
    BMI: float
    DiabetesPedigreeFunction: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    Age: float
    Pregnancies: float

@app.post("/predict")
async def predict(data: PatientData):
    if not ARTIFACTS:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    
    try:
        # Convert input to DataFrame
        input_data = data.dict()
        df = pd.DataFrame([input_data])
        
        # Ensure columns are in correct order
        df = df[ARTIFACTS["feature_names"]]
        
        # Preprocessing
        # 1. Handle zeros (same as training)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
             if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        
        # 2. Impute
        df_imp = ARTIFACTS["imputer"].transform(df)
        
        # 3. Scale
        df_scaled = ARTIFACTS["scaler"].transform(df_imp)
        
        # Predict
        prediction = ARTIFACTS["model"].predict(df_scaled)
        prob = ARTIFACTS["model"].predict_proba(df_scaled)[:, 1]
        
        return {
            "risk_category": int(prediction[0]),
            "risk_probability": float(prob[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health():
    if not ARTIFACTS:
        raise HTTPException(status_code=503, detail="Unhealthy: Artifacts missing")
    return {"status": "healthy"}
