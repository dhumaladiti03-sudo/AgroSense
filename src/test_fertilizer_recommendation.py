
from fertilizer_recommendation import (
    get_fertilizer_recommendation,
    get_top_crops_fertilizer
)


print("\n======================================")
print("   AGROSENSE FERTILIZER RECOMMENDATION")
print("======================================")

# Test one crop
result = get_fertilizer_recommendation("rice")

if result:
    print("\nCrop:", result["crop"].title())
    print("Nutrient Focus:", result["nutrient_focus"])
    print("Fertilizer Options:", result["fertilizer_options"])
    print("Application Guidance:", result["application_guidance"])
    print("Source:", result["source"])

# Test top 3 crops
print("\n======================================")
print("          TOP 3 CROP FERTILIZERS")
print("======================================")

top_crops = [
    {"crop": "rice"},
    {"crop": "jute"},
    {"crop": "papaya"}
]

recommendations = get_top_crops_fertilizer(top_crops)

for item in recommendations:

    print("\nCrop:", item["crop"].title())
    print("Fertilizers:", item["fertilizer_options"])
    print("Guidance:", item["application_guidance"])
