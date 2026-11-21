
import joblib
import pandas as pd

from pathlib import Path


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "crop_recommendation_model.pkl"
)


# ==========================================================
# LOAD CROP MODEL
# ==========================================================

def load_crop_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Crop recommendation model not found: {MODEL_PATH}"
        )

    model_package = joblib.load(MODEL_PATH)

    model = model_package["model"]
    features = model_package["features"]

    return model, features


# ==========================================================
# TOP 3 CROP PREDICTION
# ==========================================================

def predict_top_crops(soil_parameters):

    model, features = load_crop_model()

    # Convert input parameters into DataFrame
    input_data = pd.DataFrame(
        [soil_parameters]
    )

    # Match the training feature order
    input_data = input_data[features]

    # Get prediction probabilities
    probabilities = model.predict_proba(
        input_data
    )[0]

    # Get indexes of top 3 crops
    top_indices = probabilities.argsort()[-3:][::-1]

    # Prepare results
    recommendations = []

    for index in top_indices:

        crop_name = model.classes_[index]

        confidence = float(
            probabilities[index] * 100
        )

        recommendations.append({
            "crop": crop_name,
            "confidence": round(confidence, 2)
        })

    return recommendations
