import pandas as pd

# Load dataset
df = pd.read_csv("data/soil_health/Crop_recommendation.csv")

# Parameters we will use
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

print("\n--- PARAMETER STATISTICS ---")
print(df[features].describe().T)

print("\n--- MINIMUM VALUES ---")
print(df[features].min())

print("\n--- MAXIMUM VALUES ---")
print(df[features].max())

print("\n--- UNIQUE CROPS ---")
print(df["label"].nunique())

print("\n--- CROP NAMES ---")
print(sorted(df["label"].unique()))