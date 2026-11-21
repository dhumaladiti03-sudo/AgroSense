
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "soil_health" / "soil_health_dataset.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "soil_health_model.pkl"

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\nLoading Soil Health Dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# --------------------------------------------------

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

TARGET = "soil_health"

X = df[FEATURES]
y = df[TARGET]

print("\nInput features:")
print(FEATURES)

print("\nTarget categories:")
print(y.value_counts())

# --------------------------------------------------
# 4. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 5. DEFINE ML MODELS
# --------------------------------------------------

models = {

    "Random Forest": Pipeline([
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Extra Trees": Pipeline([
        ("classifier", ExtraTreesClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("classifier", GradientBoostingClassifier(
            random_state=42
        ))
    ]),

    "Hist Gradient Boosting": Pipeline([
        ("classifier", HistGradientBoostingClassifier(
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced"
        ))
    ]),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])
}

# --------------------------------------------------
# 6. CROSS-VALIDATION AND MODEL COMPARISON
# --------------------------------------------------

print("\nTraining and comparing models...\n")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
trained_models = {}

for name, model in models.items():

    print(f"Training {name}...")

    # Cross-validation on training data
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1
    )

    # Train model on complete training split
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "CV F1 Mean": cv_scores.mean(),
        "CV F1 Std": cv_scores.std(),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

# --------------------------------------------------
# 7. DISPLAY COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV F1 Mean",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "CV F1 Mean": "{:.4f}".format,
            "CV F1 Std": "{:.4f}".format,
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

# --------------------------------------------------
# 8. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# --------------------------------------------------
# 9. FINAL EVALUATION
# --------------------------------------------------

final_predictions = best_model.predict(X_test)

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )
)

print("\n========== CONFUSION MATRIX ==========\n")

print(
    confusion_matrix(
        y_test,
        final_predictions,
        labels=["Poor", "Moderate", "Good"]
    )
)

# --------------------------------------------------
# 10. SAVE BEST MODEL
# --------------------------------------------------

model_package = {
    "model": best_model,
    "features": FEATURES,
    "classes": list(best_model.classes_),
    "best_model_name": best_model_name,
    "evaluation": results_df.to_dict(orient="records")
}

joblib.dump(model_package, MODEL_FILE)

# Save comparison report
REPORT_FILE = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(REPORT_FILE, index=False)

print("\n========== COMPLETED ==========")

print("Best model:", best_model_name)

print("Model saved at:", MODEL_FILE)

print("Comparison report saved at:", REPORT_FILE)

print("\nSoil Health Model Training Successful!")
