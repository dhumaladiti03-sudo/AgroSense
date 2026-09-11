import pandas as pd

# Load dataset
df = pd.read_csv("data/soil_health/Crop_recommendation.csv")

# Display basic information
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DATA TYPES ---")
print(df.dtypes)