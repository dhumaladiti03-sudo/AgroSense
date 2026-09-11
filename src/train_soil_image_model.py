import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# SETTINGS
# ==========================================

train_dir = "data/soil_images/processed/train"
val_dir = "data/soil_images/processed/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# First training phase
INITIAL_EPOCHS = 12

# Fine-tuning phase
FINE_TUNE_EPOCHS = 15

model_path = "models/soil_image_model.keras"
class_names_path = "models/soil_image_classes.json"

# ==========================================
# START
# ==========================================

print("\n===================================")
print("IMPROVED SOIL IMAGE MODEL TRAINING")
print("===================================")

print("TensorFlow:", tf.__version__)

# ==========================================
# LOAD DATASET
# ==========================================

train_dataset = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_dataset = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "->", name)

# Save class names
os.makedirs("models", exist_ok=True)

with open(class_names_path, "w") as file:
    json.dump(class_names, file)

# ==========================================
# CALCULATE CLASS WEIGHTS
# ==========================================

class_counts = []

for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)

    count = len([
        file for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ])

    class_counts.append(count)

print("\nTraining image counts:")
for name, count in zip(class_names, class_counts):
    print(f"{name}: {count}")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.concatenate([
        np.full(count, i)
        for i, count in enumerate(class_counts)
    ])
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
for i, weight in class_weights.items():
    print(f"{class_names[i]}: {weight:.3f}")

# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.20),
    layers.RandomContrast(0.15),
])

# ==========================================
# LOAD MOBILENETV2
# ==========================================

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model initially
base_model.trainable = False

# ==========================================
# BUILD MODEL
# ==========================================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# CALLBACKS
# ==========================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
]

# ==========================================
# PHASE 1
# ==========================================

print("\n===================================")
print("PHASE 1: TRANSFER LEARNING")
print("===================================")

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# PHASE 2: FINE-TUNING
# ==========================================

print("\n===================================")
print("PHASE 2: FINE-TUNING")
print("===================================")

# Unfreeze MobileNetV2
base_model.trainable = True

# Keep early layers frozen
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with smaller learning rate
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# FINAL EVALUATION
# ==========================================

print("\n===================================")
print("FINAL MODEL EVALUATION")
print("===================================")

loss, accuracy = model.evaluate(val_dataset)

print("\nFinal Validation Loss:", loss)
print("Final Validation Accuracy:", accuracy)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(model_path)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")

print("Model:")
print(model_path)

print("\nClasses:")
print(class_names_path)

print("\n===================================")
print("IMPROVED TRAINING COMPLETE!")
print("===================================")