
from combined_crop_prediction import predict_combined_crops

# Seven soil parameters
soil_parameters = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 202.9
}

# Example soil type from the image classifier
soil_type = "Alluvial_Soil"


print("\n======================================")
print(" AGROSENSE COMBINED CROP PREDICTION")
print("======================================")

print("\nSoil Type:", soil_type)

print("\nTop 3 Recommended Crops")
print("--------------------------------------")

results = predict_combined_crops(
    soil_parameters,
    soil_type
)

for rank, result in enumerate(results, start=1):

    print(f"\nRank {rank}: {result['crop'].title()}")
    print(
        "Model Probability:",
        f"{result['model_probability']}%"
    )
    print(
        "Soil Suitability:",
        result["soil_suitability"]
    )
    print(
        "Soil Type:",
        result["soil_type"]
    )

print("\n======================================")
print("       PREDICTION COMPLETED")
print("======================================")
