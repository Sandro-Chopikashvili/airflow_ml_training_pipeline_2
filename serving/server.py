from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

model = None

mlflow.set_tracking_uri("http://mlflow:5000")

def get_model():
    global model
    if model is None:
        try:
            pyfunc_model = mlflow.pyfunc.load_model("models:/credit_card_model_classification@champion")
            model = pyfunc_model.unwrap_python_model() 
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not ready: {str(e)}")
    return model  


class PredictRequest(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    m = get_model()
    df = pd.DataFrame([request.model_dump()])
    
    try:
        proba = m.predict_proba(df)
        probability = float(proba[0][1])
        return {
            "default_probability": round(probability, 4),
            "prediction": 1 if probability >= 0.5 else 0,
            "risk_level": "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_batch(requests: list[PredictRequest]):
    m = get_model()
    df = pd.DataFrame([r.model_dump() for r in requests])
    
    try:
        probas = m.predict_proba(df)
        return {
            "predictions": [
                {
                    "default_probability": round(float(p), 4),
                    "prediction": 1 if p >= 0.5 else 0,
                    "risk_level": "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"
                }
                for p in probas[:, 1]
            ],
            "total": len(probas)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")