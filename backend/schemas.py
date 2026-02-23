from pydantic import BaseModel

class PatientFeatures(BaseModel):
    avg_heart_rate: float
    max_heart_rate: float
    min_spo2: float
    sbp_std: float
    hr_trend: float
