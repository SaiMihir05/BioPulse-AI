import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

print("🚑 BioPulse ICU Ingestion Started")

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

# 1. Load data
df = pd.read_csv("data/sample_icu.csv")
print("📥 Raw data shape:", df.shape)
print("📥 Raw columns:", df.columns.tolist())

# 🔒 SAFETY CHECK (REAL ENGINEERING)
required_cols = {"patient_id", "timestamp", "heart_rate", "sbp", "dbp", "spo2"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

# 2. Cleaning
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["patient_id", "timestamp"])
# Forward fill vitals ONLY (not patient_id)
vital_cols = ["heart_rate", "sbp", "dbp", "spo2"]
df[vital_cols] = (
    df.groupby("patient_id")[vital_cols]
      .ffill()
)

# patient_id is NEVER touched now
print("🧱 Columns before DB write:", df.columns.tolist())
print(df.head())

# 3. Store in PostgreSQL
df.to_sql(
    name="icu_vitals",
    con=engine,
    if_exists="replace",
    index=False
)

print("✅ ICU data ingested into PostgreSQL (table: icu_vitals)")
