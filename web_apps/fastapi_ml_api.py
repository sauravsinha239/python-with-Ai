"""
FastAPI Machine Learning API

A modern REST API for serving machine learning models using FastAPI.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import uvicorn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="A REST API for machine learning model predictions",
    version="1.0.0"
)

# Pydantic models for request/response
class PredictionInput(BaseModel):
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.5, -0.3, 1.2, -0.8]
            }
        }

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    confidence: str

class ModelInfo(BaseModel):
    model_type: str
    n_features: int
    n_classes: int
    classes: List[int]
    status: str

# Global variables
model = None
scaler = None

def train_model():
    """Train and save the model."""
    global model, scaler
    
    # Create dataset
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, 
                              random_state=42, n_redundant=0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Create directory and save
    os.makedirs('web_apps', exist_ok=True)
    joblib.dump(model, 'web_apps/fastapi_model.pkl')
    joblib.dump(scaler, 'web_apps/fastapi_scaler.pkl')
    
    return model, scaler

def load_model():
    """Load the trained model."""
    global model, scaler
    
    try:
        model = joblib.load('web_apps/fastapi_model.pkl')
        scaler = joblib.load('web_apps/fastapi_scaler.pkl')
        return True
    except FileNotFoundError:
        model, scaler = train_model()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when the API starts."""
    if not load_model():
        raise Exception("Failed to load machine learning model")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "model_info": "/model-info",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction using the loaded model."""
    try:
        features = input_data.features
        
        # Validate input
        if len(features) != 4:
            raise HTTPException(status_code=400, detail="Expected 4 features")
        
        # Convert and scale features
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        max_probability = probabilities.max()
        
        # Determine confidence level
        if max_probability > 0.8:
            confidence = "high"
        elif max_probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionOutput(
            prediction=int(prediction),
            probability=float(max_probability),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    try:
        return ModelInfo(
            model_type=type(model).__name__,
            n_features=int(model.n_features_in_),
            n_classes=len(model.classes_),
            classes=model.classes_.tolist(),
            status="active"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None and scaler is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/batch-predict")
async def batch_predict(features_batch: List[List[float]]):
    """Make predictions for multiple samples."""
    try:
        if not features_batch:
            raise HTTPException(status_code=400, detail="No features provided")
        
        # Validate all samples have 4 features
        for i, features in enumerate(features_batch):
            if len(features) != 4:
                raise HTTPException(status_code=400, detail=f"Sample {i}: Expected 4 features")
        
        # Convert to numpy array and scale
        features_array = np.array(features_batch)
        features_scaled = scaler.transform(features_array)
        
        # Make predictions
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            max_prob = probs.max()
            confidence = "high" if max_prob > 0.8 else "medium" if max_prob > 0.6 else "low"
            
            results.append({
                "sample_id": i,
                "prediction": int(pred),
                "probability": float(max_prob),
                "confidence": confidence
            })
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("=== FastAPI ML Model Server ===")
    print("Starting server...")
    print("API documentation will be available at: http://127.0.0.1:8000/docs")
    print("Interactive API explorer at: http://127.0.0.1:8000/redoc")
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)