import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("🧠 BioPulse ML Training Started")

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

# 1. Load features
df = pd.read_sql("SELECT * FROM patient_features", engine)
print("📥 Loaded feature table:", df.shape)

# 2. Simulate risk label (TEMPORARY)
# High risk if:
# - heart rate trend is high OR
# - minimum SpO2 is low
df["risk"] = ((df["hr_trend"] > 10) | (df["min_spo2"] < 95)).astype(int)

print("🏷️ Risk label distribution:")
print(df["risk"].value_counts())

# 3. Prepare X, y
X = df.drop(columns=["patient_id", "risk"])
y = df["risk"]

# ⚠️ Small dataset: train on full data (prototype phase)
X_train, y_train = X, y


# 5. Train model
model = LogisticRegression()
model.fit(X_train, y_train)



# 7. Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/risk_model.joblib")

print("✅ Model trained and saved (models/risk_model.joblib)")
