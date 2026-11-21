
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score
)


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "soil_health"
    / "Crop_recommendation.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "crop_recommendation_model.pkl"


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

TARGET = "label"


# ==========================================================
# LOAD DATASET
# ==========================================================

print("\n==============================================")
print("       AGROSENSE CROP MODEL TRAINING")
print("==============================================")

print("\nLoading crop recommendation dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

data = pd.read_csv(DATASET_PATH)

print("Dataset shape:", data.shape)


# ==========================================================
# DATA VALIDATION
# ==========================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

data = data[required_columns].copy()

for column in FEATURES:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data = data.dropna()

data[TARGET] = data[TARGET].astype(str).str.strip()

data = data[data[TARGET] != ""]

print("\nClean dataset shape:", data.shape)

print("\nNumber of crop categories:", data[TARGET].nunique())

print("\nCrop distribution:")
print(data[TARGET].value_counts())


# ==========================================================
# PREPARE INPUTS AND OUTPUTS
# ==========================================================

X = data[FEATURES]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================================
# TRAIN RANDOM FOREST MODEL
# ==========================================================

print("\nTraining Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================================
# MODEL EVALUATION
# ==========================================================

print("\n==============================================")
print("             MODEL EVALUATION")
print("==============================================")

y_pred = model.predict(X_test)

probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)

top3_accuracy = top_k_accuracy_score(
    y_test,
    probabilities,
    k=3,
    labels=model.classes_
)

print(f"\nTop-1 Accuracy: {accuracy * 100:.2f}%")
print(f"Top-3 Accuracy: {top3_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))


# ==========================================================
# SAVE MODEL
# ==========================================================

model_package = {
    "model": model,
    "features": FEATURES,
    "classes": list(model.classes_),
    "accuracy": accuracy,
    "top3_accuracy": top3_accuracy
}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\n==============================================")
print("          MODEL SAVED SUCCESSFULLY")
print("==============================================")

print("\nSaved location:", MODEL_PATH)

print("\nCrop categories:", len(model.classes_))

print("\nAgroSense crop model training completed!")
