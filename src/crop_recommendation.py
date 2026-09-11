# ============================================
# AgroSense - Crop Recommendation Engine
# ============================================

# Crop suitability rules based on:
# Soil type + N + P + K + pH + Temperature
# + Humidity + Rainfall


CROP_RULES = {
    "Alluvial_Soil": {
        "rice":       {"N": (60, 100), "P": (30, 60), "K": (30, 60), "ph": (5.5, 7.5), "temperature": (20, 35), "humidity": (60, 90), "rainfall": (100, 250)},
        "wheat":      {"N": (50, 100), "P": (25, 60), "K": (25, 60), "ph": (6.0, 7.5), "temperature": (15, 30), "humidity": (40, 80), "rainfall": (50, 150)},
        "maize":      {"N": (50, 100), "P": (25, 60), "K": (25, 60), "ph": (5.5, 7.5), "temperature": (18, 32), "humidity": (40, 80), "rainfall": (50, 150)},
    },

    "Black_Soil": {
        "cotton":     {"N": (50, 100), "P": (20, 60), "K": (20, 60), "ph": (5.5, 8.0), "temperature": (20, 35), "humidity": (40, 80), "rainfall": (50, 150)},
        "soybean":    {"N": (40, 80), "P": (20, 60), "K": (20, 60), "ph": (6.0, 7.5), "temperature": (20, 30), "humidity": (50, 80), "rainfall": (60, 150)},
        "maize":      {"N": (50, 100), "P": (25, 60), "K": (25, 60), "ph": (5.5, 7.5), "temperature": (18, 32), "humidity": (40, 80), "rainfall": (50, 150)},
    },

    "Red_Soil": {
        "groundnut":  {"N": (20, 60), "P": (20, 50), "K": (20, 60), "ph": (5.5, 7.0), "temperature": (20, 35), "humidity": (40, 80), "rainfall": (50, 150)},
        "millet":     {"N": (20, 60), "P": (15, 50), "K": (15, 50), "ph": (5.5, 7.5), "temperature": (20, 35), "humidity": (30, 70), "rainfall": (30, 100)},
        "maize":      {"N": (50, 100), "P": (25, 60), "K": (25, 60), "ph": (5.5, 7.5), "temperature": (18, 32), "humidity": (40, 80), "rainfall": (50, 150)},
    },

    "Laterite_Soil": {
        "cashew":     {"N": (20, 60), "P": (15, 50), "K": (20, 60), "ph": (5.0, 7.0), "temperature": (20, 35), "humidity": (50, 90), "rainfall": (100, 250)},
        "tea":        {"N": (30, 80), "P": (15, 50), "K": (20, 60), "ph": (4.5, 6.5), "temperature": (15, 30), "humidity": (60, 95), "rainfall": (150, 300)},
        "coffee":     {"N": (30, 80), "P": (15, 50), "K": (20, 60), "ph": (5.0, 7.0), "temperature": (18, 30), "humidity": (60, 95), "rainfall": (100, 250)},
    },

    "Arid_Soil": {
        "millet":     {"N": (20, 60), "P": (15, 50), "K": (15, 50), "ph": (6.0, 8.0), "temperature": (25, 40), "humidity": (20, 60), "rainfall": (20, 100)},
        "chickpea":   {"N": (20, 60), "P": (20, 50), "K": (20, 60), "ph": (6.0, 8.0), "temperature": (15, 30), "humidity": (30, 70), "rainfall": (30, 100)},
        "groundnut":  {"N": (20, 60), "P": (20, 50), "K": (20, 60), "ph": (5.5, 7.0), "temperature": (20, 35), "humidity": (40, 80), "rainfall": (50, 150)},
    },

    "Yellow_Soil": {
        "rice":       {"N": (60, 100), "P": (30, 60), "K": (30, 60), "ph": (5.5, 7.5), "temperature": (20, 35), "humidity": (60, 90), "rainfall": (100, 250)},
        "maize":      {"N": (50, 100), "P": (25, 60), "K": (25, 60), "ph": (5.5, 7.5), "temperature": (18, 32), "humidity": (40, 80), "rainfall": (50, 150)},
        "groundnut":  {"N": (20, 60), "P": (20, 50), "K": (20, 60), "ph": (5.5, 7.0), "temperature": (20, 35), "humidity": (40, 80), "rainfall": (50, 150)},
    },

    "Mountain_Soil": {
        "apple":      {"N": (40, 80), "P": (20, 60), "K": (20, 60), "ph": (5.5, 7.0), "temperature": (10, 25), "humidity": (40, 80), "rainfall": (50, 200)},
        "tea":        {"N": (30, 80), "P": (15, 50), "K": (20, 60), "ph": (4.5, 6.5), "temperature": (15, 30), "humidity": (60, 95), "rainfall": (150, 300)},
        "coffee":     {"N": (30, 80), "P": (15, 50), "K": (20, 60), "ph": (5.0, 7.0), "temperature": (18, 30), "humidity": (60, 95), "rainfall": (100, 250)},
    },
}


def calculate_crop_score(crop_conditions, soil_values):
    """
    Calculate suitability score for one crop.
    """

    parameters = [
        "N",
        "P",
        "K",
        "ph",
        "temperature",
        "humidity",
        "rainfall"
    ]

    matched = 0

    for parameter in parameters:

        minimum, maximum = crop_conditions[parameter]
        value = soil_values[parameter]

        if minimum <= value <= maximum:
            matched += 1

    score = (matched / len(parameters)) * 100

    return score


def recommend_crops(
    soil_type,
    N,
    P,
    K,
    ph,
    temperature,
    humidity,
    rainfall
):
    """
    Return Top 3 suitable crops for the predicted soil type.
    """

    soil_values = {
        "N": N,
        "P": P,
        "K": K,
        "ph": ph,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall
    }

    if soil_type not in CROP_RULES:
        return []

    crop_scores = []

    for crop, conditions in CROP_RULES[soil_type].items():

        score = calculate_crop_score(
            conditions,
            soil_values
        )

        crop_scores.append(
            {
                "crop": crop,
                "score": round(score, 2)
            }
        )

    # Sort crops from highest score to lowest
    crop_scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Return Top 3
    return crop_scores[:3]


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    recommendations = recommend_crops(
        soil_type="Black_Soil",
        N=80,
        P=40,
        K=40,
        ph=6.5,
        temperature=25,
        humidity=65,
        rainfall=100
    )

    print("\n==============================")
    print("TOP 3 CROP RECOMMENDATIONS")
    print("==============================")

    for i, result in enumerate(recommendations, start=1):

        print(
            f"{i}. {result['crop'].title()} "
            f"- {result['score']}% suitability"
        )