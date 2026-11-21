
from crop_prediction import predict_top_crops


# ==========================================================
# SAMPLE SOIL PARAMETERS
# ==========================================================

soil_parameters = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}


# ==========================================================
# RUN CROP PREDICTION
# ==========================================================

print("\n======================================")
print("     AGROSENSE CROP RECOMMENDATION")
print("======================================")

print("\nEntered Soil Parameters:")

for parameter, value in soil_parameters.items():
    print(f"{parameter}: {value}")


recommendations = predict_top_crops(
    soil_parameters
)


# ==========================================================
# DISPLAY TOP 3 CROPS
# ==========================================================

print("\nTOP 3 RECOMMENDED CROPS")
print("--------------------------------------")

for rank, result in enumerate(recommendations, start=1):

    print(f"\nRank {rank}: {result['crop'].title()}")

    print(
        f"Model Probability: {result['confidence']}%"
    )


print("\n======================================")
print("       PREDICTION COMPLETED")
print("======================================")
