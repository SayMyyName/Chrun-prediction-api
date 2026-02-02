from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('Production-Ready Machine Learning API using FastAPI\churn_model.pkl')
encoders = joblib.load('encoders.pkl')
feature_order = joblib.load('feature_order.pkl')

app = FastAPI(title='Customer Churn Predictor API')

class ChurnInput(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: float
    OnlineBackup: float
    DeviceProtection: float
    TechSupport: float
    StreamingTV: float
    StreamingMovies: float
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict_churn(data: ChurnInput):
    input_df = pd.DataFrame([data.model_dump()])

    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.fit_transform(input_df[col])
    
    churn_prob = model.predict_proba(input_df)[:, 1]
    churn_pred = 'Yes' if churn_prob >= 0.5 else 'No'

    return {
        'churn_probability': round(float(churn_prob), 3),
        'churn_prediction': churn_pred
    }

