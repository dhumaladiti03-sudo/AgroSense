import pandas as pd
import numpy as np

# -----------------------------------------
# 1. Load original dataset
# -----------------------------------------

input_file = "data/soil_health/Crop_recommendation.csv"

df = pd.read_csv(input_file)

# -----------------------------------------
# 2. Create parameter quality scores
# -----------------------------------------

# For N, P and K:
# Values closer to the dataset's normal range
# receive better scores.

def nutrient_score(series):
    median = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)

    if iqr == 0:
        iqr = 1

    score = 100 * np.exp(
        -0.5 * ((series - median) / iqr) ** 2
    )

    return score.clip(0, 100)


df["N_score"] = nutrient_score(df["N"])
df["P_score"] = nutrient_score(df["P"])
df["K_score"] = nutrient_score(df["K"])


# -----------------------------------------
# 3. pH score
# -----------------------------------------

# Ideal pH is taken as approximately 6.5
# for our project-defined soil health index.

ideal_ph = 6.5

df["pH_score"] = (
    100 - (abs(df["ph"] - ideal_ph) / 3.0) * 100
).clip(0, 100)


# -----------------------------------------
# 4. Temperature score
# -----------------------------------------

ideal_temperature = df["temperature"].median()

temperature_range = (
    df["temperature"].quantile(0.75)
    - df["temperature"].quantile(0.25)
)

if temperature_range == 0:
    temperature_range = 1

df["temperature_score"] = (
    100 * np.exp(
        -0.5
        * ((df["temperature"] - ideal_temperature)
           / temperature_range) ** 2
    )
).clip(0, 100)


# -----------------------------------------
# 5. Humidity score
# -----------------------------------------

ideal_humidity = df["humidity"].median()

humidity_range = (
    df["humidity"].quantile(0.75)
    - df["humidity"].quantile(0.25)
)

if humidity_range == 0:
    humidity_range = 1

df["humidity_score"] = (
    100 * np.exp(
        -0.5
        * ((df["humidity"] - ideal_humidity)
           / humidity_range) ** 2
    )
).clip(0, 100)


# -----------------------------------------
# 6. Rainfall score
# -----------------------------------------

ideal_rainfall = df["rainfall"].median()

rainfall_range = (
    df["rainfall"].quantile(0.75)
    - df["rainfall"].quantile(0.25)
)

if rainfall_range == 0:
    rainfall_range = 1

df["rainfall_score"] = (
    100 * np.exp(
        -0.5
        * ((df["rainfall"] - ideal_rainfall)
           / rainfall_range) ** 2
    )
).clip(0, 100)


# -----------------------------------------
# 7. Calculate raw soil health index
# -----------------------------------------

df["raw_score"] = (
    df["N_score"] * 0.20
    + df["P_score"] * 0.15
    + df["K_score"] * 0.15
    + df["pH_score"] * 0.25
    + df["temperature_score"] * 0.10
    + df["humidity_score"] * 0.07
    + df["rainfall_score"] * 0.08
)


# -----------------------------------------
# 8. Convert raw score into 0-100
# -----------------------------------------

# Percentile normalization spreads the dataset
# across the complete 0-100 range.

df["soil_health_score"] = (
    df["raw_score"].rank(pct=True) * 100
)


# -----------------------------------------
# 9. Assign soil health category
# -----------------------------------------

def health_category(score):

    if score <= 35:
        return "Poor"

    elif score <= 69:
        return "Moderate"

    else:
        return "Good"


df["soil_health"] = df["soil_health_score"].apply(
    health_category
)


# -----------------------------------------
# 10. Remove temporary columns
# -----------------------------------------

columns_to_remove = [
    "N_score",
    "P_score",
    "K_score",
    "pH_score",
    "temperature_score",
    "humidity_score",
    "rainfall_score",
    "raw_score"
]

df = df.drop(columns=columns_to_remove)


# -----------------------------------------
# 11. Save final dataset
# -----------------------------------------

output_file = "data/soil_health/soil_health_dataset.csv"

df.to_csv(output_file, index=False)


# -----------------------------------------
# 12. Display results
# -----------------------------------------

print("\n--- FIRST 10 RECORDS ---")
print(df.head(10))

print("\n--- SOIL HEALTH DISTRIBUTION ---")
print(df["soil_health"].value_counts())

print("\n--- SCORE STATISTICS ---")
print(df["soil_health_score"].describe())

print("\n--- MINIMUM SCORE ---")
print(df["soil_health_score"].min())

print("\n--- MAXIMUM SCORE ---")
print(df["soil_health_score"].max())

print("\nDataset successfully saved to:")
print(output_file)