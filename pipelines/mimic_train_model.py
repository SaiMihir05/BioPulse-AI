import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("🧬 Training Real ICU Mortality Model")

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL)

df = pd.read_sql("SELECT * FROM mimic_modeling_dataset", engine)

print("Dataset shape:", df.shape)
print("Mortality distribution:")
print(df["hospital_expire_flag"].value_counts())

X = df.drop(columns=["stay_id", "hospital_expire_flag"])
# Handle missing values (median imputation)


y = df["hospital_expire_flag"]
X = X.fillna(X.median())

# Proper train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("🎯 ROC-AUC:", roc_auc_score(y_test, y_prob))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/mimic_mortality_model.joblib")

print("✅ Real mortality model trained and saved")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - ICU Mortality Model")
plt.savefig("models/roc_curve.png")

print(" ROC curve saved to models/roc_curve.png")