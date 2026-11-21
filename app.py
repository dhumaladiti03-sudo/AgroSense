import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from PIL import Image

from src.nutrient_analysis import analyze_nutrient_deficiency
from src.soil_image_prediction import predict_soil_type
from src.combined_crop_prediction import predict_combined_crops

try:
    from src.fertilizer_recommendation import get_fertilizer_recommendation
except ImportError:
    # Fallback so the app still runs if the fertilizer module is missing.
    def get_fertilizer_recommendation(crop):
        return None


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AgroSense | Smart Soil Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# PATHS AND MODEL LOADING
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "soil_health_model.pkl"
CSS_PATH = BASE_DIR / "style.css"


@st.cache_resource
def load_soil_model(path):
    try:
        model_package = joblib.load(path)
        return model_package["model"], model_package["features"], None
    except Exception as e:
        return None, None, str(e)


def load_css(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )


soil_health_model, MODEL_FEATURES, MODEL_LOAD_ERROR = load_soil_model(MODEL_PATH)

load_css(CSS_PATH)


# ==========================================================
# SESSION STATE
# ==========================================================

for key, default in {
    "page": "Home",
    "soil_inputs": None,
    "soil_health": None,
    "soil_image_result": None,
    "soil_image_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================================
# HELPERS
# ==========================================================

def go_to(page):
    st.session_state.page = page
    st.rerun()


def section_header(title, description):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-description">{description}</div>',
        unsafe_allow_html=True
    )


def footer():
    st.markdown(
        '<div class="custom-footer">'
        'AgroSense © 2026 | Smart Soil Intelligence'
        '</div>',
        unsafe_allow_html=True
    )


def nav_button(label, page):
    """Sidebar button. The active page is drawn as a primary (dark green) button."""
    is_active = st.session_state.page == page

    return st.button(
        label,
        key=f"nav_{page}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    )


SUITABILITY_BADGE = {
    "Suitable": "badge-suitable",
    "Context-dependent": "badge-context",
}


# ==========================================================
# CROP RECOMMENDATION DISPLAY
# ==========================================================

def display_crop_recommendations(soil_parameters, soil_type):

    st.markdown("---")
    st.markdown("## 🌾 Top 3 Recommended Crops")

    st.write(
        "Crop recommendations are generated using your seven "
        "numerical soil parameters. Soil image classification "
        "provides additional soil-type suitability information."
    )

    try:

        with st.spinner("Generating crop recommendations..."):
            recommendations = predict_combined_crops(
                soil_parameters,
                soil_type
            )

        if not recommendations:
            st.warning("No crop recommendations available.")
            return

        # ---------------- TOP 3 CROP CARDS ----------------

        columns = st.columns(3)

        for index, result in enumerate(recommendations[:3]):

            crop = result["crop"].title()
            probability = result["model_probability"]
            suitability = result["soil_suitability"]
            source = result["source"]

            badge_class = SUITABILITY_BADGE.get(
                suitability,
                "badge-neutral"
            )

            rank_class = "rank-1" if index == 0 else ""

            # Show suitability only when a mapping exists.
            has_mapping = (
                suitability != "Not verified"
                and source != "No mapping available"
            )

            suitability_html = (
                f'<p class="crop-label">Soil suitability</p>'
                f'<span class="badge {badge_class}">{suitability}</span>'
                f'<p class="crop-ref">Reference: {source}</p>'
            ) if has_mapping else ""

            card_html = (
                f'<div class="crop-card {rank_class}">'
                f'<p class="crop-rank">Rank {index + 1}</p>'
                f'<p class="crop-name">🌱 {crop}</p>'
                f'<p class="crop-label">Model probability</p>'
                f'<p class="crop-value">{probability:.2f}%</p>'
                f'{suitability_html}'
                f'</div>'
            )

            with columns[index]:

                # Crop recommendation card
                st.markdown(
                    card_html,
                    unsafe_allow_html=True
                )

                # ---------------- FERTILIZER GUIDANCE ----------------

                fertilizer = get_fertilizer_recommendation(crop)

                st.markdown("#### 🧪 Fertilizer Guidance")

                if fertilizer is None:

                    st.info(
                        "Fertilizer guidance is not available "
                        "for this crop yet."
                    )

                else:

                    st.write("**Nutrient focus:**")
                    st.write(fertilizer["nutrient_focus"])

                    st.write("**Fertilizer options:**")
                    st.write(fertilizer["fertilizer_options"])

                    st.write("**Application guidance:**")
                    st.write(fertilizer["application_guidance"])

                    st.caption(
                        f"Reference: {fertilizer['source']}"
                    )

        # ---------------- SUMMARY TABLE ----------------

        st.markdown("---")
        st.markdown("### 📋 Recommendation Summary")

        summary_df = pd.DataFrame([
            {
                "Crop": r["crop"].title(),
                "Model Probability (%)": round(
                    r["model_probability"], 2
                ),
                "Soil Type": r["soil_type"].replace("_", " "),
            }
            for r in recommendations
        ])

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Note: Model probabilities represent predictions from "
            "the numerical crop model, not guaranteed crop success. "
            "Soil suitability information is shown only where a "
            "mapping is available. Fertilizer guidance is general "
            "and does not provide exact application doses. Confirm "
            "fertilizer type and quantity using a soil test and local "
            "agricultural recommendations."
        )

    except Exception as e:
        st.error(f"Crop recommendation failed: {e}")


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

with st.sidebar:

    st.markdown(
        '<div style="padding: 10px 4px 22px;">'
        '<div style="font-size: 34px;">🌱</div>'
        '<div class="brand-name">AgroSense</div>'
        '<div class="brand-tag">Smart soil intelligence</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Menu")

    if nav_button("🏠 Home", "Home"):
        go_to("Home")

    if nav_button("🌿 Soil Assessment", "Soil Assessment"):
        go_to("Soil Assessment")

    if nav_button("🌾 Crop Recommendation", "Crop Recommendation"):

        if st.session_state.soil_health == "Poor":
            st.warning(
                "Crop recommendation is unavailable because "
                "your soil has been classified as Poor."
            )
        else:
            go_to("Crop Recommendation")

    if nav_button("ℹ️ About AgroSense", "About"):
        go_to("About")

    st.markdown("---")

    st.caption("AI-powered agricultural assistance")


# ==========================================================
# HOME PAGE
# ==========================================================

if st.session_state.page == "Home":

    st.markdown(
        '<div class="hero">'
        '<div class="hero-label">AI-powered agricultural intelligence</div>'
        '<div class="hero-title">Understand your soil.<br>Grow a better future.</div>'
        '<div class="hero-description">'
        'AgroSense uses machine learning and image classification '
        'to analyze soil health, identify soil types and recommend '
        'crops for better agricultural decisions.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    section_header(
        "What AgroSense offers",
        "An intelligent approach to understanding soil and improving crop decisions."
    )

    features = [
        ("🧪", "Soil Health Analysis",
         "Analyze N, P, K, temperature, humidity, pH and rainfall using machine learning."),
        ("📸", "Soil Image Recognition",
         "Identify soil types from uploaded soil images using image classification."),
        ("🌾", "Crop Recommendation",
         "Get crop recommendations from soil parameters with additional soil-type suitability information."),
    ]

    for column, (icon, title, description) in zip(st.columns(3), features):
        with column:
            st.markdown(
                '<div class="feature-card">'
                f'<div class="feature-icon">{icon}</div>'
                f'<div class="feature-title">{title}</div>'
                f'<div class="feature-description">{description}</div>'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="sub-title">How it works</div>', unsafe_allow_html=True)

    steps = [
        ("Enter soil details", "Add your N, P, K, temperature, humidity, pH and rainfall values."),
        ("Upload a soil photo", "Available when your soil is rated Moderate or Good."),
        ("Review your crops", "See the top 3 crops with probability and soil suitability."),
    ]

    st.markdown(
        "".join(
            f'<div class="step-row"><div class="step-num">{i}</div>'
            f'<div class="step-text"><b>{title}.</b> {text}</div></div>'
            for i, (title, text) in enumerate(steps, start=1)
        ),
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.2, 1])

    with center:
        if st.button(
            "Start Soil Assessment →",
            key="home_start",
            use_container_width=True,
            type="primary"
        ):
            go_to("Soil Assessment")

    footer()


# ==========================================================
# SOIL ASSESSMENT PAGE
# ==========================================================

elif st.session_state.page == "Soil Assessment":

    section_header(
        "Soil health assessment",
        "Enter your soil nutrient and environmental parameters. "
        "AgroSense will analyze your soil using machine learning."
    )

    if soil_health_model is None:

        st.error("Soil Health Model could not be loaded.")
        st.caption(f"Details: {MODEL_LOAD_ERROR}")

    else:

        # ---------------- NUTRIENTS ----------------

        st.markdown("### 🧪 Soil Nutrient Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            nitrogen = st.number_input(
                "Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0
            )

        with col2:
            phosphorus = st.number_input(
                "Phosphorus (P)", min_value=5.0, max_value=145.0, value=50.0
            )

        with col3:
            potassium = st.number_input(
                "Potassium (K)", min_value=5.0, max_value=205.0, value=50.0
            )

        # ---------------- ENVIRONMENT ----------------

        st.markdown("### 🌤️ Environmental Parameters")

        col4, col5 = st.columns(2)

        with col4:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-10.0, max_value=60.0, value=25.0, step=0.5
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0, max_value=100.0, value=60.0, step=1.0
            )

        with col5:
            ph = st.number_input(
                "Soil pH",
                min_value=0.0, max_value=14.0, value=6.5, step=0.1
            )

            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0, max_value=1000.0, value=100.0, step=1.0
            )

        st.markdown("---")

        # ---------------- PREDICTION ----------------

        if st.button(
            "Analyze Soil Health",
            key="analyze_soil",
            use_container_width=True,
            type="primary"
        ):

            try:

                input_data = pd.DataFrame(
                    [[nitrogen, phosphorus, potassium,
                      temperature, humidity, ph, rainfall]],
                    columns=["N", "P", "K", "temperature",
                             "humidity", "ph", "rainfall"]
                )

                input_data = input_data[MODEL_FEATURES]

                prediction = soil_health_model.predict(input_data)[0]

                st.session_state.soil_inputs = input_data.iloc[0].to_dict()
                st.session_state.soil_health = prediction
                st.session_state.soil_image_result = None
                st.session_state.soil_image_id = None

                st.rerun()

            except Exception as e:
                st.error(f"An error occurred during soil prediction: {e}")

        # ---------------- SAVED RESULT ----------------

        if st.session_state.soil_health is not None:

            prediction = st.session_state.soil_health
            saved_inputs = st.session_state.soil_inputs
            input_data = pd.DataFrame([saved_inputs])

            st.markdown("---")

            st.markdown(
                '<div class="result-title">🌱 Soil Health Assessment Result</div>',
                unsafe_allow_html=True
            )

            # ---- POOR ----

            if prediction == "Poor":

                st.error("Soil Health: POOR")

                st.warning(
                    "Your soil has been classified as Poor. "
                    "Nutrient deficiencies may need attention."
                )

                st.markdown("### 🧪 Nutrient Deficiency Analysis")

                deficiencies = analyze_nutrient_deficiency(
                    saved_inputs["N"],
                    saved_inputs["P"],
                    saved_inputs["K"]
                )

                for item in deficiencies:

                    st.markdown(f"#### {item['nutrient']}")

                    if item["level"] == "Low":
                        st.error(f"Status: {item['level']}")
                    else:
                        st.info(f"Status: {item['level']}")

                    st.write(f"**Possible symptoms:** {item['symptoms']}")
                    st.write(f"**Improvement guidance:** {item['solution']}")

                st.warning(
                    "Crop recommendation and soil image analysis "
                    "are unavailable for this Poor soil assessment."
                )

            # ---- MODERATE ----

            elif prediction == "Moderate":

                st.warning("Soil Health: MODERATE")

                st.info(
                    "Your soil has been classified as Moderate. "
                    "Further analysis can help identify improvement opportunities."
                )

            # ---- GOOD ----

            elif prediction == "Good":

                st.success("Soil Health: GOOD")

                st.info(
                    "Your soil has been classified as Good "
                    "by the trained model."
                )

            else:
                st.warning(f"Predicted category: {prediction}")

            # ---- ENTERED PARAMETERS ----

            st.markdown(
                '<div class="sub-title">Entered soil parameters</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

            # ==============================================
            # SOIL IMAGE CLASSIFICATION
            # ==============================================

            if prediction in ["Moderate", "Good"]:

                st.markdown("---")

                st.markdown(
                    '<div class="result-title">📸 Soil Image Classification</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    "Upload a clear image of your soil to identify "
                    "its soil category using the trained image model."
                )

                uploaded_image = st.file_uploader(
                    "Upload Soil Image",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="soil_image_upload"
                )

                if uploaded_image is not None:

                    image = Image.open(uploaded_image).convert("RGB")

                    st.image(
                        image,
                        caption="Uploaded Soil Image",
                        width=400
                    )

                    image_id = (uploaded_image.name, uploaded_image.size)

                    if st.button(
                        "Identify Soil Type",
                        key="identify_soil",
                        use_container_width=True,
                        type="primary"
                    ):

                        try:

                            with st.spinner("Analyzing soil image..."):
                                result = predict_soil_type(image)
                                st.session_state.soil_image_result = result
                                st.session_state.soil_image_id = image_id

                        except Exception as e:
                            st.error(f"Soil image prediction failed: {e}")

                    # ---- IMAGE PREDICTION RESULT ----

                    if (
                        st.session_state.soil_image_result is not None
                        and st.session_state.soil_image_id == image_id
                    ):

                        result = st.session_state.soil_image_result
                        soil_type = result["soil_type"]
                        confidence = result["confidence"]

                        st.markdown("---")

                        st.markdown("### 🌱 Soil Classification Result")

                        st.success(
                            f"Predicted Soil Type: "
                            f"{soil_type.replace('_', ' ')}"
                        )

                        st.metric(
                            "Prediction Confidence",
                            f"{confidence:.2f}%"
                        )

                        if confidence < 60:
                            st.warning(
                                "The model has low confidence in this prediction. "
                                "Try uploading a clearer soil image."
                            )
                        elif confidence < 80:
                            st.info(
                                "The prediction has moderate confidence. "
                                "Please verify it with a physical soil assessment."
                            )
                        else:
                            st.success(
                                "The model has high confidence in this prediction. "
                                "This is still an automated estimate, not a laboratory result."
                            )

                        st.markdown("#### Class-wise Prediction Scores")

                        prediction_data = pd.DataFrame([
                            {
                                "Soil Type": name.replace("_", " "),
                                "Confidence (%)": score
                            }
                            for name, score in result["all_predictions"].items()
                        ]).sort_values(by="Confidence (%)", ascending=False)

                        st.bar_chart(
                            prediction_data.set_index("Soil Type"),
                            color="#39734D"
                        )

                        display_crop_recommendations(saved_inputs, soil_type)

                else:

                    st.info(
                        "Upload a soil image to continue with "
                        "soil classification and crop recommendations."
                    )


# ==========================================================
# CROP RECOMMENDATION PAGE
# ==========================================================

elif st.session_state.page == "Crop Recommendation":

    section_header(
        "Crop recommendation",
        "View crop predictions based on your soil parameters "
        "and the available soil-type suitability information."
    )

    if st.session_state.soil_health == "Poor":

        st.error("Crop recommendation is unavailable for Poor soil.")

    elif st.session_state.soil_health in ["Moderate", "Good"]:

        if st.session_state.soil_image_result is None:

            st.info("Complete soil image classification first.")

        elif st.session_state.soil_inputs is None:

            st.info("Please complete soil parameter assessment first.")

        else:

            soil_type = st.session_state.soil_image_result["soil_type"]

            st.success("Soil image classification completed.")

            st.write(
                f"**Predicted soil type:** {soil_type.replace('_', ' ')}"
            )

            display_crop_recommendations(
                st.session_state.soil_inputs,
                soil_type
            )

    else:

        st.info("Please complete Soil Health Assessment first.")


# ==========================================================
# ABOUT PAGE
# ==========================================================

elif st.session_state.page == "About":

    section_header(
        "About AgroSense",
        "An AI-based approach to soil analysis and agricultural assistance."
    )

    st.markdown("### 🌱 Our Project")

    st.write(
        "AgroSense is a machine learning and image classification "
        "based agricultural assistance system designed to analyze "
        "soil parameters, identify soil types and support crop decisions."
    )

    st.markdown("### 🧠 Technologies Used")

    st.write(
        "- Python\n"
        "- Streamlit\n"
        "- Pandas and NumPy\n"
        "- Scikit-learn\n"
        "- TensorFlow and Keras\n"
        "- MobileNetV2"
    )

    st.markdown("### 🎯 Project Objective")

    st.write(
        "To provide an accessible prototype for soil health assessment "
        "and image-based soil classification with crop recommendation support."
    )

    footer()