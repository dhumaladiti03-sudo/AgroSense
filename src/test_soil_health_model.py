import joblib
import numpy as np
import pandas as pd

# -----------------------------------------
# 1. Load model and scaler
# -----------------------------------------

model = joblib.load("models/soil_health_model.pkl")
scaler = joblib.load("models/soil_health_scaler.pkl")

# -----------------------------------------
# 2. Load original data
#    Used for calculating the same
#    project-defined score as Step 8
# -----------------------------------------

df = pd.read_csv(
    "data/soil_health/Crop_recommendation.csv"
)


# -----------------------------------------
# 3. Score calculation functions
# -----------------------------------------

def nutrient_score(value, series):

    median = series.median()

    iqr = (
        series.quantile(0.75)
        - series.quantile(0.25)
    )

    if iqr == 0:
        iqr = 1

    score = 100 * np.exp(
        -0.5 * ((value - median) / iqr) ** 2
    )

    return float(np.clip(score, 0, 100))


def parameter_score(value, series):

    median = series.median()

    iqr = (
        series.quantile(0.75)
        - series.quantile(0.25)
    )

    if iqr == 0:
        iqr = 1

    score = 100 * np.exp(
        -0.5 * ((value - median) / iqr) ** 2
    )

    return float(np.clip(score, 0, 100))


def ph_score(value):

    ideal_ph = 6.5

    score = (
        100
        - (abs(value - ideal_ph) / 3.0) * 100
    )

    return float(np.clip(score, 0, 100))


# -----------------------------------------
# 4. Enter soil parameters
# -----------------------------------------

print("\n===================================")
print("       AGROSENSE SOIL HEALTH")
print("===================================")

N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))

temperature = float(
    input("Temperature (°C): ")
)

humidity = float(
    input("Humidity (%): ")
)

ph = float(
    input("pH: ")
)

rainfall = float(
    input("Rainfall (mm): ")
)


# -----------------------------------------
# 5. Calculate individual scores
# -----------------------------------------

N_score = nutrient_score(N, df["N"])
P_score = nutrient_score(P, df["P"])
K_score = nutrient_score(K, df["K"])

temperature_score = parameter_score(
    temperature,
    df["temperature"]
)

humidity_score = parameter_score(
    humidity,
    df["humidity"]
)

rainfall_score = parameter_score(
    rainfall,
    df["rainfall"]
)

pH_score_value = ph_score(ph)


# -----------------------------------------
# 6. Calculate raw score
# -----------------------------------------

raw_score = (
    N_score * 0.20
    + P_score * 0.15
    + K_score * 0.15
    + pH_score_value * 0.25
    + temperature_score * 0.10
    + humidity_score * 0.07
    + rainfall_score * 0.08
)


# -----------------------------------------
# 7. Convert raw score to approximate
#    project score
# -----------------------------------------

# The score is converted to a 0–100
# scale for display.

soil_health_score = np.clip(
    raw_score,
    0,
    100
)


# -----------------------------------------
# 8. Prepare input for SVM
# -----------------------------------------

input_data = np.array([[
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
]])

input_scaled = scaler.transform(input_data)


# -----------------------------------------
# 9. Predict health category
# -----------------------------------------

prediction = model.predict(input_scaled)[0]


# -----------------------------------------
# 10. Display result
# -----------------------------------------

print("\n===================================")
print("       SOIL HEALTH RESULT")
print("===================================")

print(
    f"Soil Health Score: "
    f"{soil_health_score:.2f} / 100"
)

print(
    f"Soil Health: {prediction}"
)


# -----------------------------------------
# 11. Recommendation
# -----------------------------------------

if prediction == "Poor":

    print("\n⚠️ Soil health is POOR.")
    print("Crop prediction will NOT be performed.")
    print("Please improve the soil and test again.")

elif prediction == "Moderate":

    print("\n🟡 Soil health is MODERATE.")
    print("You can continue to soil image analysis.")

elif prediction == "Good":

    print("\n🟢 Soil health is GOOD.")
    print("You can continue to soil image analysis.")


print("\n===================================")
print("TEST COMPLETED")
print("===================================")