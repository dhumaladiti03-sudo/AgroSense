
# --------------------------------------------------
# AGROSENSE - NUTRIENT DEFICIENCY ANALYSIS
# --------------------------------------------------

def analyze_nutrient_deficiency(nitrogen, phosphorus, potassium):

    deficiencies = []

    # Nitrogen deficiency
    if nitrogen < 40:
        deficiencies.append({
            "nutrient": "Nitrogen (N)",
            "level": "Low",
            "symptoms": "Slow plant growth and yellowing of older leaves.",
            "solution": "Consider nitrogen-rich organic matter or a suitable nitrogen fertilizer."
        })

    # Phosphorus deficiency
    if phosphorus < 25:
        deficiencies.append({
            "nutrient": "Phosphorus (P)",
            "level": "Low",
            "symptoms": "Stunted growth and possible dark green or purplish leaves.",
            "solution": "Consider a suitable phosphorus source after checking soil pH and soil-test results."
        })

    # Potassium deficiency
    if potassium < 25:
        deficiencies.append({
            "nutrient": "Potassium (K)",
            "level": "Low",
            "symptoms": "Yellowing or browning along leaf edges and weak stems.",
            "solution": "Consider a suitable potassium source based on soil-test recommendations."
        })

    if not deficiencies:
        deficiencies.append({
            "nutrient": "No specific NPK deficiency detected",
            "level": "Not identified",
            "symptoms": "The selected NPK values did not cross the project's low-value thresholds.",
            "solution": "Get a soil test before applying fertilizer."
        })

    return deficiencies
