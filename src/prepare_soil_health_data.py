import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------------------
# 1. Load soil health dataset
# -----------------------------------------

file_path = "data/soil_health/soil_health_dataset.csv"

df = pd.read_csv(file_path)

print("\n--- DATASET LOADED ---")
print("Shape:", df.shape)


# -----------------------------------------
# 2. Select input features
# -----------------------------------------

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = df[features]

# Target
y = df["soil_health"]


# -----------------------------------------
# 3. Display features and target
# -----------------------------------------

print("\n--- INPUT FEATURES ---")
print(X.head())

print("\n--- TARGET ---")
print(y.head())


# -----------------------------------------
# 4. Split dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 5. Display split sizes
# -----------------------------------------

print("\n--- TRAINING DATA ---")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n--- TESTING DATA ---")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# -----------------------------------------
# 6. Feature scaling
# -----------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# -----------------------------------------
# 7. Save scaler
# -----------------------------------------

scaler_path = "models/soil_health_scaler.pkl"

joblib.dump(scaler, scaler_path)

print("\nScaler saved successfully:")
print(scaler_path)


# -----------------------------------------
# 8. Save prepared datasets
# -----------------------------------------

train_data = pd.DataFrame(
    X_train_scaled,
    columns=features
)

train_data["soil_health"] = y_train.values

test_data = pd.DataFrame(
    X_test_scaled,
    columns=features
)

test_data["soil_health"] = y_test.values


train_data.to_csv(
    "data/soil_health/train_soil_health.csv",
    index=False
)

test_data.to_csv(
    "data/soil_health/test_soil_health.csv",
    index=False
)


print("\nTraining dataset saved:")
print("data/soil_health/train_soil_health.csv")

print("\nTesting dataset saved:")
print("data/soil_health/test_soil_health.csv")


# -----------------------------------------
# 9. Final confirmation
# -----------------------------------------

print("\n===================================")
print("DATA PREPARATION COMPLETED!")
print("===================================")