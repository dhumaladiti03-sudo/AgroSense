
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "soil_images" / "split"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42

MODEL_PATH = MODEL_DIR / "soil_image_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "soil_image_classes.json"

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

print("\nLoading Soil Image Dataset...\n")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "val",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR / "test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nSoil Classes:")
for name in class_names:
    print("-", name)

print("\nNumber of soil classes:", num_classes)

# Check that all splits contain the same class order
if val_ds.class_names != class_names or test_ds.class_names != class_names:
    raise ValueError("Class names or class order differ between dataset splits.")

# Save class names for later predictions
with open(CLASS_NAMES_PATH, "w") as file:
    json.dump(class_names, file, indent=4)

# --------------------------------------------------
# CALCULATE CLASS WEIGHTS
# --------------------------------------------------

print("\nCalculating Class Weights...")

train_labels = np.concatenate([
    labels.numpy() for _, labels in train_ds
])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}

print("\nClass Weights:")
for index, weight in class_weights.items():
    print(f"{class_names[index]}: {weight:.3f}")

# --------------------------------------------------
# OPTIMIZE DATA PIPELINE
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10)
], name="data_augmentation")

# --------------------------------------------------
# LOAD MOBILENETV2
# --------------------------------------------------

print("\nLoading MobileNetV2 Transfer Learning Model...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

# --------------------------------------------------
# BUILD CLASSIFICATION MODEL
# --------------------------------------------------

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = layers.Lambda(preprocess_input)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="AgroSense_Soil_Classifier"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Architecture:")
model.summary()

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001
    )
]

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nStarting Soil Image Model Training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# --------------------------------------------------
# LOAD BEST MODEL
# --------------------------------------------------

print("\nLoading Best Model...")

best_model = tf.keras.models.load_model(MODEL_PATH)

# --------------------------------------------------
# TEST MODEL
# --------------------------------------------------

print("\nEvaluating on Test Dataset...\n")

test_loss, test_accuracy = best_model.evaluate(test_ds)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print("\nGenerating Classification Report...")

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

y_probabilities = best_model.predict(test_ds)

y_pred = np.argmax(y_probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:\n")
print(report)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
)

print("\nConfusion Matrix:\n")
print(matrix)

# Save evaluation results
evaluation = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "class_names": class_names
}

with open(MODEL_DIR / "soil_image_evaluation.json", "w") as file:
    json.dump(evaluation, file, indent=4)

# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print("\n" + "=" * 55)
print("AGROSENSE SOIL IMAGE MODEL TRAINING COMPLETED")
print("=" * 55)

print("\nBest model saved at:")
print(MODEL_PATH)

print("\nClass names saved at:")
print(CLASS_NAMES_PATH)

print("\nEvaluation saved at:")
print(MODEL_DIR / "soil_image_evaluation.json")

print("\nSoil Image Classification Model Ready!")
