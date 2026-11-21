
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "soil_health" / "Crop_recommendation.csv"

OUTPUT_FILE = BASE_DIR / "data" / "soil_health" / "soil_health_dataset.csv"

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading crop recommendation dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())

# --------------------------------------------------
# DATA CLEANING
# --------------------------------------------------

required_columns = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall", "label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Convert numerical columns to numeric values
numeric_columns = [
    "N", "P", "K",
    "temperature", "humidity",
    "ph", "rainfall"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing or invalid values
df = df.dropna(subset=required_columns).copy()

# Remove impossible values
valid_rows = (
    (df["N"] >= 0) &
    (df["P"] >= 0) &
    (df["K"] >= 0) &
    (df["temperature"] >= -10) &
    (df["temperature"] <= 60) &
    (df["humidity"] >= 0) &
    (df["humidity"] <= 100) &
    (df["ph"] >= 0) &
    (df["ph"] <= 14) &
    (df["rainfall"] >= 0)
)

df = df.loc[valid_rows].copy()

print("\nDataset shape after cleaning:", df.shape)

# --------------------------------------------------
# NORMALIZE N, P AND K
# --------------------------------------------------

def min_max_normalize(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(0.5, index=series.index)

    return (series - minimum) / (maximum - minimum)


df["N_score"] = min_max_normalize(df["N"])
df["P_score"] = min_max_normalize(df["P"])
df["K_score"] = min_max_normalize(df["K"])

# --------------------------------------------------
# CALCULATE OTHER PARAMETER SCORES
# --------------------------------------------------

df["pH_score"] = np.clip(
    1 - abs(df["ph"] - 6.5) / 3.5,
    0, 1
)

df["temperature_score"] = np.clip(
    1 - abs(df["temperature"] - 25) / 20,
    0, 1
)

df["humidity_score"] = np.clip(
    1 - abs(df["humidity"] - 70) / 60,
    0, 1
)

df["rainfall_score"] = np.clip(
    1 - abs(df["rainfall"] - 100) / 200,
    0, 1
)

# --------------------------------------------------
# WEIGHTED SOIL HEALTH SCORE
# --------------------------------------------------

df["raw_score"] = (
    df["N_score"] * 20 +
    df["P_score"] * 15 +
    df["K_score"] * 15 +
    df["pH_score"] * 25 +
    df["temperature_score"] * 10 +
    df["humidity_score"] * 7 +
    df["rainfall_score"] * 8
)

# --------------------------------------------------
# RESCALE SCORE FROM 0 TO 100
# --------------------------------------------------

df["soil_health_score"] = (
    df["raw_score"].rank(pct=True) * 100
)

df["soil_health_score"] = df["soil_health_score"].clip(0, 100)

# --------------------------------------------------
# CLASSIFY SOIL HEALTH
# --------------------------------------------------

def classify_soil(score):

    if score <= 35:
        return "Poor"

    elif score <= 69:
        return "Moderate"

    else:
        return "Good"


df["soil_health"] = df["soil_health_score"].apply(classify_soil)

# --------------------------------------------------
# REMOVE TEMPORARY COLUMNS
# --------------------------------------------------

temporary_columns = [
    "N_score",
    "P_score",
    "K_score",
    "pH_score",
    "temperature_score",
    "humidity_score",
    "rainfall_score",
    "raw_score"
]

df.drop(columns=temporary_columns, inplace=True)

# --------------------------------------------------
# SAVE FINAL DATASET
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nSOIL HEALTH DATASET CREATED SUCCESSFULLY!")

print("\nFinal dataset shape:")
print(df.shape)

print("\nSoil health category distribution:")
print(df["soil_health"].value_counts())

print("\nSoil health score statistics:")
print(df["soil_health_score"].describe())

print("\nFinal dataset columns:")
print(df.columns.tolist())

print("\nSaved file:")
print(OUTPUT_FILE)
