
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# FILE PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FERTILIZER_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "soil_health"
    / "crop_fertilizer_guidance.csv"
)


# --------------------------------------------------
# LOAD FERTILIZER DATASET
# --------------------------------------------------

def load_fertilizer_data():

    if not FERTILIZER_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fertilizer dataset not found: {FERTILIZER_DATA_PATH}"
        )

    data = pd.read_csv(FERTILIZER_DATA_PATH)

    required_columns = [
        "crop",
        "nutrient_focus",
        "fertilizer_options",
        "application_guidance",
        "source"
    ]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    data["crop"] = data["crop"].astype(str).str.strip().str.lower()

    return data


# --------------------------------------------------
# GET FERTILIZER RECOMMENDATION FOR A CROP
# --------------------------------------------------

def get_fertilizer_recommendation(crop_name):

    data = load_fertilizer_data()

    crop_name = str(crop_name).strip().lower()

    result = data[data["crop"] == crop_name]

    if result.empty:
        return None

    recommendation = result.iloc[0]

    return {
        "crop": recommendation["crop"],
        "nutrient_focus": recommendation["nutrient_focus"],
        "fertilizer_options": recommendation["fertilizer_options"],
        "application_guidance": recommendation["application_guidance"],
        "source": recommendation["source"]
    }


# --------------------------------------------------
# GET FERTILIZER RECOMMENDATIONS FOR TOP CROPS
# --------------------------------------------------

def get_top_crops_fertilizer(top_crops):

    recommendations = []

    for crop_info in top_crops:

        crop_name = crop_info["crop"]

        fertilizer_info = get_fertilizer_recommendation(
            crop_name
        )

        if fertilizer_info is not None:
            recommendations.append(fertilizer_info)

    return recommendations
