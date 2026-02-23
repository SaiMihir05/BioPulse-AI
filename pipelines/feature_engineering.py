import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

print("🧠 Feature engineering started")

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

# Load ICU data
df = pd.read_sql("SELECT * FROM icu_vitals", engine)
print("📥 Loaded ICU data:", df.shape)

# Feature engineering per patient
features = []

for pid, group in df.groupby("patient_id"):
    group = group.sort_values("timestamp")

    feat = {
        "patient_id": pid,
        "avg_heart_rate": group["heart_rate"].mean(),
        "max_heart_rate": group["heart_rate"].max(),
        "min_spo2": group["spo2"].min(),
        "sbp_std": group["sbp"].std(),
        "hr_trend": group["heart_rate"].iloc[-1] - group["heart_rate"].iloc[0]
    }

    features.append(feat)

features_df = pd.DataFrame(features)

print("🧪 Feature table preview:")
print(features_df)

# Store features
features_df.to_sql(
    name="patient_features",
    con=engine,
    if_exists="replace",
    index=False
)

print("✅ Patient features stored (table: patient_features)")
