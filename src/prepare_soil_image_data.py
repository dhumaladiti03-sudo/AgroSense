import os
import shutil
import random

# Original dataset location
source_dir = "data/soil_images/Orignal-Dataset"

# New dataset location
output_dir = "data/soil_images/processed"

# Train-validation split
train_ratio = 0.80

# Make results reproducible
random.seed(42)

# Create output folders
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("\n===================================")
print("PREPARING SOIL IMAGE DATASET")
print("===================================")

# Process every soil class
for class_name in sorted(os.listdir(source_dir)):

    class_source = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_source):
        continue

    # Get image files
    images = [
        file for file in os.listdir(class_source)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    # Shuffle images
    random.shuffle(images)

    # Calculate training images
    train_count = int(len(images) * train_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:]

    # Create class folders
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)

    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_val_dir, exist_ok=True)

    # Copy training images
    for image in train_images:
        source_path = os.path.join(class_source, image)
        destination_path = os.path.join(class_train_dir, image)
        shutil.copy2(source_path, destination_path)

    # Copy validation images
    for image in val_images:
        source_path = os.path.join(class_source, image)
        destination_path = os.path.join(class_val_dir, image)
        shutil.copy2(source_path, destination_path)

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} validation"
    )

print("-----------------------------------")
print("IMAGE DATASET PREPARATION COMPLETE")
print("===================================")