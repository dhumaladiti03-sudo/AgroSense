
import random
import shutil
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "soil_images" / "cleaned"
OUTPUT_DIR = BASE_DIR / "data" / "soil_images" / "split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

random.seed(RANDOM_SEED)


# --------------------------------------------------
# SPLIT DATASET
# --------------------------------------------------

def main():

    print("\nStarting Soil Image Dataset Splitting...\n")

    if not SOURCE_DIR.exists():
        print("ERROR: Cleaned dataset folder not found.")
        return

    if not abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9:
        print("ERROR: Split ratios must add up to 1.")
        return

    # Create output folders
    for split_name in ["train", "val", "test"]:
        (OUTPUT_DIR / split_name).mkdir(
            parents=True,
            exist_ok=True
        )

    summary = defaultdict(dict)

    total_train = 0
    total_val = 0
    total_test = 0

    # Process each soil class
    for class_folder in sorted(SOURCE_DIR.iterdir()):

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        images = [
            path for path in class_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        ]

        images.sort()

        if len(images) < 3:
            print(f"Skipping {class_name}: not enough images.")
            continue

        # Shuffle images reproducibly
        random.shuffle(images)

        total_images = len(images)

        # Calculate split counts
        train_count = int(total_images * TRAIN_RATIO)
        val_count = int(total_images * VAL_RATIO)

        test_count = total_images - train_count - val_count

        # Ensure all three splits contain images
        if train_count < 1 or val_count < 1 or test_count < 1:
            print(f"Skipping {class_name}: split would be empty.")
            continue

        train_images = images[:train_count]

        val_images = images[
            train_count:train_count + val_count
        ]

        test_images = images[
            train_count + val_count:
        ]

        # Copy images into respective folders
        split_data = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        for split_name, split_images in split_data.items():

            destination_folder = (
                OUTPUT_DIR / split_name / class_name
            )

            destination_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            for image_path in split_images:

                shutil.copy2(
                    image_path,
                    destination_folder / image_path.name
                )

        # Save counts
        summary[class_name] = {
            "total": total_images,
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images)
        }

        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)

    # --------------------------------------------------
    # DISPLAY REPORT
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("SOIL IMAGE DATASET SPLITTING REPORT")
    print("=" * 65)

    print(
        f"\n{'Class':<20}"
        f"{'Total':>8}"
        f"{'Train':>8}"
        f"{'Val':>8}"
        f"{'Test':>8}"
    )

    print("-" * 65)

    for class_name, counts in sorted(summary.items()):

        print(
            f"{class_name:<20}"
            f"{counts['total']:>8}"
            f"{counts['train']:>8}"
            f"{counts['val']:>8}"
            f"{counts['test']:>8}"
        )

    total = total_train + total_val + total_test

    print("-" * 65)

    print(
        f"{'TOTAL':<20}"
        f"{total:>8}"
        f"{total_train:>8}"
        f"{total_val:>8}"
        f"{total_test:>8}"
    )

    if total > 0:
        print("\nActual split percentages:")

        print(f"Training:   {total_train / total * 100:.2f}%")
        print(f"Validation: {total_val / total * 100:.2f}%")
        print(f"Testing:    {total_test / total * 100:.2f}%")

    print("\nDataset saved at:")
    print(OUTPUT_DIR)

    print("\nDataset splitting completed successfully!")


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()
