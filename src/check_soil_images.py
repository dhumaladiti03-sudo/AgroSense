import os

dataset_path = "data/soil_images/Orignal-Dataset"
print("\n===================================")
print("SOIL IMAGE DATASET CHECK")
print("===================================")

total_images = 0

for folder in sorted(os.listdir(dataset_path)):

    folder_path = os.path.join(
        dataset_path,
        folder
    )

    if os.path.isdir(folder_path):

        images = [
            file for file in os.listdir(folder_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        ]

        count = len(images)
        total_images += count

        print(f"{folder}: {count} images")


print("-----------------------------------")
print(f"Total images: {total_images}")
print("===================================")