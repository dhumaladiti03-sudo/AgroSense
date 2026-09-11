import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------------
# 1. Load prepared datasets
# -----------------------------------------

train_file = "data/soil_health/train_soil_health.csv"
test_file = "data/soil_health/test_soil_health.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X_train = train_df[features]
y_train = train_df["soil_health"]

X_test = test_df[features]
y_test = test_df["soil_health"]


print("\n===================================")
print("SOIL HEALTH ML MODEL TRAINING")
print("===================================")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------------
# 2. Define ML models
# -----------------------------------------

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    )
}


# -----------------------------------------
# 3. Train and evaluate models
# -----------------------------------------

results = []

trained_models = {}

for name, model in models.items():

    print("\n-----------------------------------")
    print("Training:", name)
    print("-----------------------------------")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    trained_models[name] = model

    print("Accuracy:", round(accuracy * 100, 2), "%")


# -----------------------------------------
# 4. Compare models
# -----------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(results_df.to_string(index=False))


# -----------------------------------------
# 5. Select best model
# -----------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_accuracy = results_df.iloc[0]["Accuracy"]


print("\n===================================")
print("BEST MODEL")
print("===================================")

print("Model:", best_model_name)
print("Accuracy:", round(best_accuracy * 100, 2), "%")


# -----------------------------------------
# 6. Detailed report
# -----------------------------------------

best_predictions = best_model.predict(X_test)

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_test,
        best_predictions
    )
)


# -----------------------------------------
# 7. Save best model
# -----------------------------------------

model_path = "models/soil_health_model.pkl"

joblib.dump(
    best_model,
    model_path
)

print("\nBest model saved successfully:")
print(model_path)


# -----------------------------------------
# 8. Save comparison results
# -----------------------------------------

results_df.to_csv(
    "models/soil_health_model_comparison.csv",
    index=False
)

print("\nModel comparison saved:")
print("models/soil_health_model_comparison.csv")


print("\n===================================")
print("STEP 10 COMPLETED!")
print("===================================")