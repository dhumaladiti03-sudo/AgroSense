
import pandas as pd
from pathlib import Path

try:
    from .crop_prediction import predict_top_crops
except ImportError:
    from crop_prediction import predict_top_crops
BASE_DIR = Path(__file__).resolve().parent.parent

MAPPING_PATH = (
    BASE_DIR
    / "data"
    / "soil_health"
    / "soil_type_crop_mapping.csv"
)


def load_soil_crop_mapping():
    """Load soil-type and crop suitability information."""

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Soil crop mapping not found: {MAPPING_PATH}"
        )

    mapping = pd.read_csv(MAPPING_PATH)

    required_columns = {
        "soil_type",
        "crop",
        "suitability",
        "source"
    }

    if not required_columns.issubset(mapping.columns):
        raise ValueError("Invalid soil crop mapping columns.")

    return mapping


def get_soil_suitability(soil_type, crop, mapping):
    """Find the available suitability information."""

    matching_rows = mapping[
        (mapping["soil_type"].str.lower() == soil_type.lower())
        & (mapping["crop"].str.lower() == crop.lower())
    ]

    if matching_rows.empty:
        return {
            "suitability": "Not verified",
            "source": "No mapping available"
        }

    row = matching_rows.iloc[0]

    return {
        "suitability": row["suitability"],
        "source": row["source"]
    }


def predict_combined_crops(soil_parameters, soil_type):
    """
    Combine numerical crop predictions with
    soil-image classification information.

    The original crop model remains unchanged.
    """

    if not isinstance(soil_parameters, dict):
        raise TypeError("Soil parameters must be a dictionary.")

    required_features = [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]

    missing_features = [
        feature
        for feature in required_features
        if feature not in soil_parameters
    ]

    if missing_features:
        raise ValueError(
            f"Missing soil parameters: {missing_features}"
        )

    if not isinstance(soil_type, str) or not soil_type.strip():
        raise ValueError("A valid soil type is required.")

    mapping = load_soil_crop_mapping()

    # Get predictions from the existing numerical model.
    numerical_predictions = predict_top_crops(
        soil_parameters
    )

    combined_results = []

    for prediction in numerical_predictions:

        crop_name = prediction["crop"]

        soil_information = get_soil_suitability(
            soil_type,
            crop_name,
            mapping
        )

        combined_results.append({
            "crop": crop_name,
            "model_probability": prediction["confidence"],
            "soil_type": soil_type,
            "soil_suitability": soil_information["suitability"],
            "source": soil_information["source"]
        })

    return combined_results
