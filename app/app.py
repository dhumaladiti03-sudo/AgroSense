import base64
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# TensorFlow
# ---------------------------------------------------------

try:
    import tensorflow as tf
    TENSORFLOW_IMPORT_ERROR = None
except Exception as e:
    tf = None
    TENSORFLOW_IMPORT_ERROR = str(e)

# ---------------------------------------------------------
# Crop Recommendation Module
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from crop_recommendation import recommend_crops
    CROP_MODULE_ERROR = None
except Exception as e:
    recommend_crops = None
    CROP_MODULE_ERROR = str(e)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="AgroSense",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Session State (single source of truth, with defaults)
# ---------------------------------------------------------

DEFAULT_STATE = {
    "start": False,
    "soil_result": None,
    "soil_parameters": None,
    "predicted_soil_type": None,
    "image_confidence": None,
    "crop_results": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_BASE_DIRS = [
    THIS_FILE_DIR,
    os.path.dirname(THIS_FILE_DIR),
    os.path.dirname(os.path.dirname(THIS_FILE_DIR)),
]


def _find_models_dir(candidates):
    for base in candidates:
        candidate_models_dir = os.path.join(base, "models")
        if os.path.isdir(candidate_models_dir):
            return candidate_models_dir
    return os.path.join(candidates[-1], "models")


MODELS_DIR = _find_models_dir(CANDIDATE_BASE_DIRS)

MODEL_PATH = os.path.join(MODELS_DIR, "soil_health_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "soil_health_scaler.pkl")
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "soil_image_model.keras")
IMAGE_CLASSES_PATH = os.path.join(MODELS_DIR, "soil_image_classes.json")
BACKGROUND_PATH = os.path.join(THIS_FILE_DIR, "background.jpg")

# ---------------------------------------------------------
# Load Soil Health Model
# ---------------------------------------------------------

soil_health_model = None
soil_health_scaler = None
soil_model_error = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"soil_health_model.pkl not found:\n{MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"soil_health_scaler.pkl not found:\n{SCALER_PATH}")

    soil_health_model = joblib.load(MODEL_PATH)
    soil_health_scaler = joblib.load(SCALER_PATH)
except Exception as e:
    soil_model_error = str(e)

# ---------------------------------------------------------
# Load Soil Image Model
# ---------------------------------------------------------

soil_image_model = None
soil_image_classes = None
image_model_error = None

try:
    if TENSORFLOW_IMPORT_ERROR:
        raise ImportError(
            "TensorFlow is not installed.\n"
            "Run: pip install tensorflow\n"
            f"Original error: {TENSORFLOW_IMPORT_ERROR}"
        )
    if not os.path.exists(IMAGE_MODEL_PATH):
        raise FileNotFoundError(f"soil_image_model.keras not found:\n{IMAGE_MODEL_PATH}")
    if not os.path.exists(IMAGE_CLASSES_PATH):
        raise FileNotFoundError(f"soil_image_classes.json not found:\n{IMAGE_CLASSES_PATH}")

    soil_image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)

    with open(IMAGE_CLASSES_PATH, "r") as file:
        soil_image_classes = json.load(file)
except Exception as e:
    image_model_error = str(e)

# ---------------------------------------------------------
# Background Image
# ---------------------------------------------------------

encoded_image = ""

try:
    if os.path.exists(BACKGROUND_PATH):
        with open(BACKGROUND_PATH, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode()
except Exception:
    encoded_image = ""

if encoded_image:
    background_css = f"""
        background-image:
            linear-gradient(
                rgba(0, 0, 0, 0.55),
                rgba(0, 0, 0, 0.55)
            ),
            url("data:image/jpeg;base64,{encoded_image}");
    """
else:
    background_css = """
        background:
            linear-gradient(
                135deg,
                #1b5e20,
                #388e3c
            );
    """

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown(
    f"""
    <style>

    .stApp {{
        {background_css}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    .hero {{
        min-height: 58vh;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding-top: 30px;
    }}

    .hero-card {{
        padding: 55px 75px;
        max-width: 850px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.30);
        border-radius: 28px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.40);
        color: white;
    }}

    .leaf {{
        font-size: 50px;
        margin-bottom: 5px;
    }}

    .logo {{
        font-size: 65px;
        font-weight: 800;
        letter-spacing: 6px;
        text-shadow: 2px 3px 10px rgba(0, 0, 0, 0.5);
    }}

    .tagline {{
        font-size: 27px;
        font-weight: 500;
        margin-top: 12px;
    }}

    .subtitle {{
        font-size: 18px;
        margin-top: 18px;
        opacity: 0.9;
    }}

    .hero-container-btn {{
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }}

    .hero-container-btn div.stButton > button {{
        width: 280px;
        height: 55px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 700;
        border: 2px solid white !important;
        background: rgba(255, 255, 255, 0.18) !important;
        color: white !important;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }}

    .hero-container-btn div.stButton > button:hover {{
        background: white !important;
        color: #1b5e20 !important;
        transform: scale(1.04);
        border-color: white !important;
    }}

    .section-card {{
        background: rgba(255, 255, 255, 0.95);
        padding: 35px;
        border-radius: 25px;
        color: #222;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }}

    .health-card {{
        padding: 30px;
        border-radius: 22px;
        margin-top: 20px;
        color: #222;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18);
    }}

    .crop-card {{
        background: rgba(255, 255, 255, 0.96);
        padding: 25px;
        border-radius: 20px;
        margin-top: 15px;
        color: #222;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18);
        text-align: center;
    }}

    .crop-rank {{
        font-size: 30px;
        font-weight: 800;
        color: #1b5e20;
    }}

    .crop-name {{
        font-size: 25px;
        font-weight: 700;
        margin-top: 8px;
    }}

    .crop-score {{
        font-size: 20px;
        color: #2e7d32;
        font-weight: 600;
        margin-top: 8px;
    }}

    label {{
        font-weight: 600 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Hero Section
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-card">
            <div class="leaf">🌱</div>
            <div class="logo">AGROSENSE</div>
            <div class="tagline">Smart Soil. Better Crops. Better Farming.</div>
            <div class="subtitle">AI-Powered Soil Health Assessment &amp; Crop Recommendation</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


def start_assessment():
    st.session_state["start"] = True


st.markdown('<div class="hero-container-btn">', unsafe_allow_html=True)
st.button("🧪 Check Your Soil →", use_container_width=False, on_click=start_assessment)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Soil Image Prediction Function
# ---------------------------------------------------------

def predict_soil_type(uploaded_file):
    uploaded_file.seek(0)

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    # IMPORTANT:
    # MobileNetV2 preprocessing is already included
    # inside the trained model.
    # Therefore, DO NOT call preprocess_input here.

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = soil_image_model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index]) * 100

    if isinstance(soil_image_classes, dict):
        predicted_class = soil_image_classes.get(
            str(predicted_index),
            "Unknown Soil"
        )

    elif (
        isinstance(soil_image_classes, list)
        and 0 <= predicted_index < len(soil_image_classes)
    ):
        predicted_class = soil_image_classes[predicted_index]

    else:
        predicted_class = "Unknown Soil"

    return predicted_class, confidence

# ---------------------------------------------------------
# Crop Recommendation Function
# ---------------------------------------------------------

def generate_crop_recommendations(
    soil_type, nitrogen, phosphorus, potassium,
    temperature, humidity, ph, rainfall
):
    if recommend_crops is None:
        raise ImportError(
            f"Crop recommendation module could not be loaded:\n{CROP_MODULE_ERROR}"
        )

    return recommend_crops(
        soil_type=soil_type,
        N=nitrogen,
        P=phosphorus,
        K=potassium,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall
    )


# ---------------------------------------------------------
# Render helper: crop recommendation results
# ---------------------------------------------------------

def render_crop_results(crop_results):
    if not crop_results:
        st.warning(
            "⚠️ No suitable crops were found for the given combination of "
            "soil parameters and soil type."
        )
        return

    crop_columns = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]

    for i, crop in enumerate(crop_results[:3]):
        crop_name = crop.get("crop", crop.get("name", "Unknown Crop"))
        crop_score = crop.get("score", crop.get("suitability", 0))

        try:
            crop_score = float(crop_score)
        except (TypeError, ValueError):
            crop_score = 0.0

        with crop_columns[i]:
            st.markdown(
                f"""
                <div class="crop-card">
                    <div class="crop-rank">{medals[i]}</div>
                    <div class="crop-name">{crop_name}</div>
                    <div class="crop-score">Suitability: {crop_score:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🌱 Crop recommendation generated successfully!")


# =========================================================
# MAIN APPLICATION
# =========================================================

if st.session_state["start"]:

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Soil Health Section
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="section-card">
            <h1 style="text-align:center;color:#1b5e20;margin-bottom:5px;">
                🌱 Soil Health Assessment
            </h1>
            <p style="text-align:center;font-size:16px;color:#555;">
                Enter the soil and environmental parameters
                to analyze your soil health.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if soil_model_error:
        st.error("⚠️ Soil health model could not be loaded.")
        st.code(soil_model_error)
    else:

        with st.form("soil_input_form"):

            st.markdown("### 🧪 Soil Nutrients")
            col1, col2, col3 = st.columns(3)

            with col1:
                nitrogen = st.number_input(
                    "Nitrogen (N) — mg/kg", min_value=0.0, max_value=200.0,
                    value=50.0, step=1.0
                )
            with col2:
                phosphorus = st.number_input(
                    "Phosphorus (P) — mg/kg", min_value=0.0, max_value=200.0,
                    value=40.0, step=1.0
                )
            with col3:
                potassium = st.number_input(
                    "Potassium (K) — mg/kg", min_value=0.0, max_value=200.0,
                    value=40.0, step=1.0
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌦️ Environmental Conditions")
            col4, col5 = st.columns(2)

            with col4:
                temperature = st.number_input(
                    "Temperature — °C", min_value=0.0, max_value=50.0,
                    value=25.0, step=0.1
                )
            with col5:
                humidity = st.number_input(
                    "Humidity — %", min_value=0.0, max_value=100.0,
                    value=65.0, step=0.1
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌧️ Soil & Rainfall")
            col6, col7 = st.columns(2)

            with col6:
                ph = st.number_input(
                    "Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1
                )
            with col7:
                rainfall = st.number_input(
                    "Rainfall — mm", min_value=0.0, max_value=500.0,
                    value=100.0, step=1.0
                )

            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button(
                "🔍 Analyze Soil Health", use_container_width=True
            )

        # -------------------------------------------------
        # Soil Health Prediction
        # -------------------------------------------------

        if submit_btn:
            try:
                input_data = pd.DataFrame(
                    [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
                    columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
                )

                scaled_data = soil_health_scaler.transform(input_data)
                prediction = str(soil_health_model.predict(scaled_data)[0]).strip()
                normalized_prediction = prediction.capitalize()

                st.session_state["soil_result"] = normalized_prediction
                st.session_state["soil_parameters"] = {
                    "N": nitrogen, "P": phosphorus, "K": potassium,
                    "temperature": temperature, "humidity": humidity,
                    "ph": ph, "rainfall": rainfall
                }

                # Reset downstream results since soil parameters changed
                st.session_state["predicted_soil_type"] = None
                st.session_state["image_confidence"] = None
                st.session_state["crop_results"] = None

            except Exception as e:
                st.session_state["soil_result"] = None
                st.error("❌ Error while analyzing soil health.")
                st.code(str(e))

        # -------------------------------------------------
        # Display Soil Health Result (persists across reruns)
        # -------------------------------------------------

        if st.session_state["soil_result"]:

            result = st.session_state["soil_result"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="section-card">
                    <h2 style="text-align:center;color:#1b5e20;">
                        📊 Soil Health Assessment Result
                    </h2>
                </div>
                """,
                unsafe_allow_html=True
            )

            if result.lower() == "poor":
                st.markdown(
                    """
                    <div class="health-card" style="background:rgba(255,235,235,0.97);border-left:7px solid #d32f2f;">
                        <h1 style="color:#b71c1c;">🔴 Poor Soil Health</h1>
                        <p>The entered soil parameters indicate that the soil health is currently poor.</p>
                        <h3>💡 Recommended Improvement Steps</h3>
                        <ul>
                            <li>Check and improve nutrient levels.</li>
                            <li>Add suitable organic matter or compost.</li>
                            <li>Maintain an appropriate soil pH.</li>
                            <li>Improve soil moisture management.</li>
                            <li>Retest the soil after improvement.</li>
                        </ul>
                        <h3 style="color:#b71c1c;">🚫 Crop Recommendation Unavailable</h3>
                        <p>Crop recommendation is available only for Moderate or Good soil health.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif result.lower() == "moderate":
                st.markdown(
                    """
                    <div class="health-card" style="background:rgba(255,248,220,0.97);border-left:7px solid #f9a825;">
                        <h1 style="color:#e65100;">🟡 Moderate Soil Health</h1>
                        <p>Your soil has moderate health. You can continue to soil image analysis.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.info("📷 Scroll down to upload your soil image.")

            elif result.lower() == "good":
                st.markdown(
                    """
                    <div class="health-card" style="background:rgba(235,250,235,0.97);border-left:7px solid #2e7d32;">
                        <h1 style="color:#1b5e20;">🟢 Good Soil Health</h1>
                        <p>Excellent! Your soil has good health. You can continue to soil image analysis.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.success("🌱 Your soil is ready for crop recommendation!")

            else:
                st.warning(f"⚠️ Unexpected model result: {result}")

    # =====================================================
    # SOIL IMAGE SECTION
    # =====================================================

    if st.session_state.get("soil_result") in ["Moderate", "Good"]:

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-card">
                <h1 style="text-align:center;color:#1b5e20;">📷 Soil Image Analysis</h1>
                <p style="text-align:center;color:#555;">
                    Upload a clear image of your soil.
                    AgroSense will analyze the visual characteristics of the soil.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        if image_model_error:
            st.error("⚠️ Soil image model could not be loaded.")
            st.code(image_model_error)
        else:

            uploaded_image = st.file_uploader(
                "📤 Upload Soil Image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear soil photograph for AI-based soil classification."
            )

            if uploaded_image is not None:

                st.markdown("### 🖼️ Uploaded Soil Image")
                uploaded_image.seek(0)
                preview_image = Image.open(uploaded_image)

                image_col1, image_col2, image_col3 = st.columns([1, 2, 1])
                with image_col2:
                    st.image(preview_image, caption="Uploaded Soil Image", use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("🤖 Analyze Soil Image", use_container_width=True):
                    try:
                        with st.spinner("AI is analyzing your soil image..."):
                            predicted_soil, confidence = predict_soil_type(uploaded_image)

                        st.session_state["predicted_soil_type"] = predicted_soil
                        st.session_state["image_confidence"] = confidence

                        # Compute crop recommendations right away and store them too
                        if CROP_MODULE_ERROR is None and st.session_state["soil_parameters"]:
                            params = st.session_state["soil_parameters"]
                            st.session_state["crop_results"] = generate_crop_recommendations(
                                predicted_soil,
                                params["N"], params["P"], params["K"],
                                params["temperature"], params["humidity"],
                                params["ph"], params["rainfall"]
                            )
                        else:
                            st.session_state["crop_results"] = None

                    except Exception as e:
                        st.session_state["predicted_soil_type"] = None
                        st.session_state["image_confidence"] = None
                        st.session_state["crop_results"] = None
                        st.error("❌ Error while analyzing the soil image or generating crop recommendations.")
                        st.code(str(e))

            # ---------------------------------------------
            # Display image + crop results (persists across reruns)
            # ---------------------------------------------

            if st.session_state["predicted_soil_type"]:

                predicted_soil = st.session_state["predicted_soil_type"]
                confidence = st.session_state["image_confidence"] or 0.0

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="section-card">
                        <h2 style="text-align:center;color:#1b5e20;">🌱 Soil Type Prediction</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                result_col1, result_col2 = st.columns(2)
                with result_col1:
                    st.metric("🌱 Predicted Soil Type", str(predicted_soil).replace("_", " "))
                with result_col2:
                    st.metric("🎯 Model Confidence", f"{confidence:.2f}%")

                st.success("✅ Soil image analysis completed!")

                if confidence >= 70:
                    st.success("🟢 High confidence prediction")
                elif confidence >= 50:
                    st.warning("🟡 Moderate confidence prediction")
                else:
                    st.warning(
                        "🟠 Low confidence prediction. "
                        "For better results, upload a clearer soil image."
                    )

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="section-card">
                        <h1 style="text-align:center;color:#1b5e20;">🌾 Top 3 Suitable Crops</h1>
                        <p style="text-align:center;color:#555;">
                            Based on your soil health parameters and AI-predicted soil type.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if CROP_MODULE_ERROR:
                    st.error("❌ Crop recommendation module could not be loaded.")
                    st.code(CROP_MODULE_ERROR)
                else:
                    render_crop_results(st.session_state["crop_results"])