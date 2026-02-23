from fastapi import FastAPI
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os


from backend.schemas import PatientFeatures

app = FastAPI(
    title="BioPulse AI",
    description="Clinical Risk Prediction API",
    version="0.1"
)

# Load trained model
model = joblib.load("models/mimic_mortality_model.joblib")


@app.get("/")
def root():
    return {"message": "BioPulse AI backend is running"}

@app.post("/predict")
def predict_risk(features: PatientFeatures):
    data = pd.DataFrame([features.dict()])
    risk_prob = model.predict_proba(data)[0][1]

    return {
        "risk_probability": round(float(risk_prob), 4),
        "risk_label": int(risk_prob > 0.5)
    }
load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)
@app.get("/predict/{stay_id}")
def predict_by_stay_id(stay_id: int):

    query = text("""
        SELECT *
        FROM mimic_modeling_dataset
        WHERE stay_id = :sid
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"sid": stay_id}).fetchone()

    if result is None:
        return {"error": f"Stay {stay_id} not found"}

    df = pd.DataFrame([dict(result._mapping)])

    # Drop non-feature columns
    X = df.drop(columns=["stay_id", "hospital_expire_flag"])

    # Handle missing values
    X = X.fillna(X.median())

    # Predict
    risk_prob = model.predict_proba(X)[0][1]

    # Calculate feature contributions
    coefficients = model.coef_[0]
    feature_names = X.columns

    contributions = {
        feature: float(X.iloc[0][feature] * coef)
        for feature, coef in zip(feature_names, coefficients)
    }

    # Sort by absolute contribution
    contributions = dict(
        sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )
    )

    return {
        "stay_id": stay_id,
        "risk_probability": round(float(risk_prob), 4),
        "risk_label": int(risk_prob > 0.5),
        "feature_contributions": contributions
    }