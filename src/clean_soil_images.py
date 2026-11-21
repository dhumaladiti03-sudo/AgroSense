
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "soil_images" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "soil_images" / "cleaned"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------
# GENERATE IMAGE HASH
# --------------------------------------------------

def get_file_hash(file_path):
    """Generate a unique hash for an image file."""

    md5 = hashlib.md5()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5.update(chunk)

    return md5.hexdigest()


# --------------------------------------------------
# MAIN CLEANING FUNCTION
# --------------------------------------------------

def main():

    print("\nStarting Soil Image Dataset Cleaning...\n")

    if not RAW_DIR.exists():
        print("ERROR: Raw dataset folder not found.")
        return

    # Collect image paths and hashes
    hash_groups = defaultdict(list)

    for class_folder in sorted(RAW_DIR.iterdir()):

        if not class_folder.is_dir():
            continue

        for image_path in sorted(class_folder.iterdir()):

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            image_hash = get_file_hash(image_path)

            hash_groups[image_hash].append(
                (class_folder.name, image_path)
            )

    # Prepare output folder
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    class_counts = defaultdict(int)

    same_class_duplicates = 0
    cross_class_conflicts = 0
    skipped_conflict_files = 0
    copied_images = 0

    print("Processing image duplicates...\n")

    # Process each unique image hash
    for image_hash, files in hash_groups.items():

        class_names = {class_name for class_name, _ in files}

        # Same image found in different classes
        if len(class_names) > 1:

            cross_class_conflicts += 1
            skipped_conflict_files += len(files)

            print("CROSS-CLASS CONFLICT:")
            for class_name, image_path in files:
                print(f"  {class_name}: {image_path.name}")

            continue

        # Keep one copy for same-class duplicates
        class_name = files[0][0]
        original_path = files[0][1]

        same_class_duplicates += len(files) - 1

        destination_folder = CLEANED_DIR / class_name
        destination_folder.mkdir(parents=True, exist_ok=True)

        destination_path = destination_folder / original_path.name

        # Copy without changing the original dataset
        shutil.copy2(original_path, destination_path)

        class_counts[class_name] += 1
        copied_images += 1

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("SOIL IMAGE DATASET CLEANING REPORT")
    print("=" * 50)

    print("\nImages copied per class:")

    for class_name in sorted(class_counts):
        print(f"{class_name}: {class_counts[class_name]}")

    print("\nTotal unique images copied:", copied_images)
    print("Same-class duplicate files skipped:", same_class_duplicates)
    print("Cross-class conflict groups:", cross_class_conflicts)
    print("Files skipped due to cross-class conflicts:", skipped_conflict_files)

    print("\nCleaned dataset location:")
    print(CLEANED_DIR)

    print("\nCleaning completed successfully!")


# --------------------------------------------------
# RUN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":
    main()
