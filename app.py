from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


app = FastAPI(
    title="Telecom Customer Prediction API",
    description="Predict Monthly Charges and Customer Churn",
    version="1.0"
)


# Load trained models

reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model_churn.pkl")


# Input data

class CustomerInfo(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    TotalCharges: float


# Home route

@app.get("/")
def home():

    return {
        "message": "Telecom Customer Prediction API is running"
    }


# Prediction route

@app.post("/predict")
def predict(customer_info: CustomerInfo):

    try:

        # Convert input into DataFrame

        input_data = pd.DataFrame([customer_info.model_dump()])


        # Regression prediction

        monthly_charges = reg_model.predict(input_data)[0]


        # Classification prediction

        churn = clf_model.predict(input_data)[0]


        return {

            "MonthlyCharges_Prediction": round(float(monthly_charges), 2),

            "Churn_Prediction": str(churn)

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )




        #python -m uvicorn app:app --reload 