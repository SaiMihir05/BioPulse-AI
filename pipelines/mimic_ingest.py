import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

print("🧬 MIMIC-IV Demo Ingestion Started")

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

BASE_PATH = "data/mimic-iv-clinical-database-demo-2.2"


# Load core tables
patients = pd.read_csv(f"{BASE_PATH}/hosp/patients.csv.gz")
admissions = pd.read_csv(f"{BASE_PATH}/hosp/admissions.csv.gz")
icustays = pd.read_csv(f"{BASE_PATH}/icu/icustays.csv.gz")

print("Patients:", patients.shape)
print("Admissions:", admissions.shape)
print("ICU stays:", icustays.shape)

# Store minimal tables
patients.to_sql("mimic_patients", engine, if_exists="replace", index=False)
admissions.to_sql("mimic_admissions", engine, if_exists="replace", index=False)
icustays.to_sql("mimic_icustays", engine, if_exists="replace", index=False)

print("✅ Core MIMIC tables stored in PostgreSQL")
