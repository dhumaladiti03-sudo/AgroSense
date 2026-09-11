import pandas as pd

df = pd.read_csv("data/soil_health/soil_health_dataset.csv")

print("\n--- SCORE RANGE DISTRIBUTION ---")

bins = [0, 35, 45, 55, 65, 70, 80, 90, 100]

labels = [
    "0-35",
    "36-45",
    "46-55",
    "56-65",
    "66-70",
    "71-80",
    "81-90",
    "91-100"
]

df["score_range"] = pd.cut(
    df["soil_health_score"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(df["score_range"].value_counts().sort_index())

print("\n--- HEALTH CATEGORY ---")
print(df["soil_health"].value_counts())

print("\n--- EXACT SCORE RANGE ---")
print("Minimum:", df["soil_health_score"].min())
print("Maximum:", df["soil_health_score"].max())