import os
import json
import joblib
import numpy as np
import tensorflow as tf

from PIL import Image

from crop_recommendation import recommend_crops


# ============================================
# LOAD MODELS
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

soil_health_model = joblib.load(
    os.path.join(BASE_DIR, "models", "soil_health_model.pkl")
)

soil_health_scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "soil_health_scaler.pkl")
)

soil_image_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "models", "soil_image_model.keras")
)

with open(
    os.path.join(BASE_DIR, "models", "soil_image_classes.json"),
    "r"
) as file:
    soil_classes = json.load(file)


# ============================================
# SOIL HEALTH PREDICTION
# ============================================

def predict_soil_health(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall
):

    values = np.array([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ]])

    scaled_values = soil_health_scaler.transform(values)

    prediction = soil_health_model.predict(scaled_values)[0]

    return prediction


# ============================================
# SOIL IMAGE PREDICTION
# ============================================

def predict_soil_type(image):

    image = image.resize((224, 224))

    image_array = np.array(image)

    # Handle grayscale images
    if len(image_array.shape) == 2:
        image_array = np.stack(
            (image_array,) * 3,
            axis=-1
        )

    # Handle RGBA images
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = soil_image_model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    soil_type = soil_classes[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    return soil_type, confidence


# ============================================
# COMPLETE AGROSENSE PIPELINE
# ============================================

def run_agrosense(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    soil_image
):

    # ----------------------------------------
    # STEP 1: Soil Health
    # ----------------------------------------

    soil_health = predict_soil_health(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    )

    print("\nSoil Health:", soil_health)

    # ----------------------------------------
    # STEP 2: Stop if soil is Poor
    # ----------------------------------------

    if soil_health == "Poor":

        return {
            "soil_health": soil_health,
            "soil_type": None,
            "confidence": None,
            "recommendations": []
        }

    # ----------------------------------------
    # STEP 3: Soil Image Classification
    # ----------------------------------------

    soil_type, confidence = predict_soil_type(
        soil_image
    )

    # ----------------------------------------
    # STEP 4: Crop Recommendation
    # ----------------------------------------

    recommendations = recommend_crops(
        soil_type=soil_type,
        N=N,
        P=P,
        K=K,
        ph=ph,
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall
    )

    return {
        "soil_health": soil_health,
        "soil_type": soil_type,
        "confidence": round(confidence, 2),
        "recommendations": recommendations
    }


# ============================================
# TEST
# ============================================

if __name__ == "__main__":

    print("\n===================================")
    print("       AGROSENSE PIPELINE TEST")
    print("===================================")

    # Test soil parameters
    N = 80
    P = 40
    K = 40
    temperature = 25
    humidity = 65
    ph = 6.5
    rainfall = 100

    # Use one image from validation dataset
    image_path = os.path.join(
        BASE_DIR,
        "data",
        "soil_images",
        "processed",
        "val",
        "Black_Soil"
    )

    image_files = [
        f for f in os.listdir(image_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if len(image_files) == 0:

        print("No test image found.")

    else:

        test_image_path = os.path.join(
            image_path,
            image_files[0]
        )

        image = Image.open(test_image_path).convert("RGB")

        result = run_agrosense(
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall,
            image
        )

        print("\n===================================")
        print("RESULT")
        print("===================================")

        print(
            "Soil Health:",
            result["soil_health"]
        )

        if result["soil_type"] is not None:

            print(
                "Soil Type:",
                result["soil_type"]
            )

            print(
                "Image Confidence:",
                result["confidence"],
                "%"
            )

            print("\nTop 3 Crops:")

            for i, crop in enumerate(
                result["recommendations"],
                start=1
            ):

                print(
                    f"{i}. "
                    f"{crop['crop'].title()} - "
                    f"{crop['score']}%"
                )

        else:

            print(
                "\nSoil health is Poor."
            )

            print(
                "Crop recommendation skipped."
            )

    print("\n===================================")
    print("PIPELINE TEST COMPLETE")
    print("===================================")