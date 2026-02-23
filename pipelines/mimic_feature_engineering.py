import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

print("🧬 Extracting ICU Vitals (Upgraded Features)")

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

BASE_PATH = "data/mimic-iv-clinical-database-demo-2.2"  # change if needed

# Vital sign itemids (MIMIC-IV)
# 220045 = Heart Rate
# 220179 = SBP
# 220180 = DBP
# 220277 = SpO2
VITAL_ITEMIDS = [220045, 220179, 220180, 220277]

# Load only required columns
chartevents = pd.read_csv(
    f"{BASE_PATH}/icu/chartevents.csv.gz",
    usecols=["stay_id", "itemid", "valuenum"],
    compression="gzip"
)

print("Chartevents shape:", chartevents.shape)

# Filter only vital signs
chartevents = chartevents[
    chartevents["itemid"].isin(VITAL_ITEMIDS)
]

print("Filtered vitals shape:", chartevents.shape)

# 🔥 Rich aggregation (mean, max, min, std)
vitals_df = chartevents.pivot_table(
    index="stay_id",
    columns="itemid",
    values="valuenum",
    aggfunc=["mean", "max", "min", "std"]
)

# Flatten multi-index columns
vitals_df.columns = [
    f"{stat}_{itemid}" for stat, itemid in vitals_df.columns
]

vitals_df = vitals_df.reset_index()

# Rename to meaningful clinical names
vitals_df = vitals_df.rename(columns={
    "mean_220045": "avg_heart_rate",
    "max_220045": "max_heart_rate",
    "std_220045": "hr_std",
    "mean_220179": "avg_sbp",
    "std_220179": "sbp_std",
    "mean_220180": "avg_dbp",
    "mean_220277": "avg_spo2",
    "min_220277": "min_spo2"
})

# Create shock index (clinically meaningful)
vitals_df["shock_index"] = (
    vitals_df["avg_heart_rate"] / vitals_df["avg_sbp"]
)

# Keep only selected engineered features
vitals_df = vitals_df[
    [
        "stay_id",
        "avg_heart_rate",
        "max_heart_rate",
        "hr_std",
        "avg_sbp",
        "sbp_std",
        "avg_dbp",
        "avg_spo2",
        "min_spo2",
        "shock_index"
    ]
]

print("Vitals aggregated shape:", vitals_df.shape)

# Load mortality labels
mortality = pd.read_sql(
    "SELECT stay_id, hospital_expire_flag FROM mimic_mortality",
    engine
)

# Merge vitals with mortality
final_df = vitals_df.merge(
    mortality,
    on="stay_id",
    how="inner"
)

print("Final dataset shape:", final_df.shape)
print(final_df.head())

# Store updated modeling dataset
final_df.to_sql(
    "mimic_modeling_dataset",
    engine,
    if_exists="replace",
    index=False
)

print("✅ Upgraded ICU modeling dataset created")
