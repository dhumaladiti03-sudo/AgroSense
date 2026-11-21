
from pathlib import Path
from PIL import Image

from soil_image_prediction import predict_soil_type


# --------------------------------------------------
# TEST IMAGE PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = (
    BASE_DIR
    / "data"
    / "soil_images"
    / "split"
    / "test"
)

# Select the first available test image
image_files = []

for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
    image_files.extend(TEST_IMAGE_DIR.rglob(extension))

if not image_files:
    raise FileNotFoundError(
        "No test images found in the test dataset."
    )

test_image_path = image_files[0]


# --------------------------------------------------
# RUN PREDICTION
# --------------------------------------------------

print("\n======================================")
print("       AGROSENSE SOIL IMAGE TEST")
print("======================================")

print("\nTesting image:", test_image_path.name)

with Image.open(test_image_path) as image:
    result = predict_soil_type(image)

print("\nPREDICTION RESULT")
print("--------------------------------------")

print("Predicted Soil Type:", result["soil_type"])
print("Confidence:", str(result["confidence"]) + "%")

print("\nALL CLASS PREDICTIONS")
print("--------------------------------------")

for soil, confidence in result["all_predictions"].items():
    print(f"{soil}: {confidence}%")

print("\n======================================")
print("       TEST COMPLETED")
print("======================================")
