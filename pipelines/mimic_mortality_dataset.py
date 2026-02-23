import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

print("🧬 Building Mortality Dataset")

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

# Load required tables
patients = pd.read_sql("SELECT subject_id FROM mimic_patients", engine)
admissions = pd.read_sql("""
    SELECT subject_id, hadm_id, hospital_expire_flag
    FROM mimic_admissions
""", engine)

icustays = pd.read_sql("""
    SELECT subject_id, hadm_id, stay_id
    FROM mimic_icustays
""", engine)

print("Admissions:", admissions.shape)
print("ICU stays:", icustays.shape)

# Join ICU stays with mortality outcome
mortality_df = icustays.merge(
    admissions,
    on=["subject_id", "hadm_id"],
    how="inner"
)

print("Merged dataset shape:", mortality_df.shape)
print(mortality_df.head())

# Store result
mortality_df.to_sql(
    "mimic_mortality",
    engine,
    if_exists="replace",
    index=False
)

print("✅ Mortality dataset created and stored")
