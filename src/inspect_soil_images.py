
import os
import hashlib
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "soil_images" / "raw"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# --------------------------------------------------
# INSPECTION FUNCTION
# --------------------------------------------------

def calculate_hash(image_path):
    """Calculate image hash to identify exact duplicates."""

    hasher = hashlib.md5()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def inspect_dataset():

    print("\n======================================")
    print("       AGROSENSE IMAGE INSPECTION")
    print("======================================")

    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder not found!")
        return

    class_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    total_images = 0
    corrupted_images = []
    duplicate_images = []
    seen_hashes = {}

    print("\nDATASET CLASS SUMMARY\n")

    for class_folder in class_folders:

        image_files = [
            file for file in class_folder.rglob("*")
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]

        valid_count = 0

        for image_path in image_files:

            try:
                with Image.open(image_path) as image:
                    image.verify()

                # Calculate hash only for valid images
                image_hash = calculate_hash(image_path)

                if image_hash in seen_hashes:
                    duplicate_images.append(
                        (image_path, seen_hashes[image_hash])
                    )
                else:
                    seen_hashes[image_hash] = image_path

                valid_count += 1

            except Exception as error:
                corrupted_images.append(
                    (image_path, str(error))
                )

        total_images += len(image_files)

        print(
            f"{class_folder.name}: "
            f"{valid_count} valid / {len(image_files)} total"
        )

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n======================================")
    print("              FINAL REPORT")
    print("======================================")

    print(f"Total image files: {total_images}")
    print(f"Corrupted images: {len(corrupted_images)}")
    print(f"Exact duplicate occurrences: {len(duplicate_images)}")

    if corrupted_images:

        print("\nCORRUPTED IMAGES:")

        for path, error in corrupted_images:
            print(f"{path} -> {error}")

    if duplicate_images:

        print("\nDUPLICATE IMAGES:")

        for duplicate, original in duplicate_images:
            print(f"{duplicate}")
            print(f"  Same content as: {original}")

    print("\nImage inspection completed!")


# --------------------------------------------------
# RUN SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    inspect_dataset()
