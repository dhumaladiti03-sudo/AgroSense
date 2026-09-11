import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# SETTINGS
# ==========================================

val_dir = "data/soil_images/processed/val"
model_path = "models/soil_image_model.keras"
class_names_path = "models/soil_image_classes.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# ==========================================
# LOAD MODEL
# ==========================================

print("\n===================================")
print("SOIL IMAGE MODEL EVALUATION")
print("===================================")

model = keras.models.load_model(model_path)

# Load class names
with open(class_names_path, "r") as file:
    class_names = json.load(file)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

# ==========================================
# LOAD VALIDATION DATA
# ==========================================

val_dataset = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================
# GET TRUE LABELS
# ==========================================

y_true = np.concatenate([
    labels.numpy()
    for images, labels in val_dataset
])

# ==========================================
# GET PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(val_dataset)

y_pred = np.argmax(predictions, axis=1)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")

cm = confusion_matrix(y_true, y_pred)

print(cm)

print("\nRows = Actual classes")
print("Columns = Predicted classes")

print("\n===================================")
print("EVALUATION COMPLETE")
print("===================================")