
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "soil_image_model.keras"
CLASS_PATH = BASE_DIR / "models" / "soil_image_classes.json"

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# LOAD MODEL AND CLASS NAMES
# --------------------------------------------------

def load_soil_image_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Soil image model not found: {MODEL_PATH}"
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Soil image classes file not found: {CLASS_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "preprocess_input": preprocess_input
        }
    )

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


# --------------------------------------------------
# PREPROCESS SOIL IMAGE
# --------------------------------------------------

def preprocess_soil_image(image):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# PREDICT SOIL TYPE
# --------------------------------------------------

def predict_soil_type(image):

    model, class_names = load_soil_image_model()

    processed_image = preprocess_soil_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))

    soil_type = class_names[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    return {
        "soil_type": soil_type,
        "confidence": round(confidence, 2),
        "all_predictions": {
            class_names[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(class_names))
        }
    }
