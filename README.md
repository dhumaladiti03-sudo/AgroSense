# AgroSense 🌱

### AI-Based Soil Health Assessment, Image-Based Soil Type , Crop Recommendation and Fertilizer Guidance Using Machine Learning&#x20;

AgroSense is an AI- and IoT-based agricultural support system that helps users assess soil health, identify soil types, receive crop recommendations, and understand possible nutrient deficiencies.

The application combines Machine Learning, image classification, and an ESP32-based soil moisture sensor to provide soil-related information through a user-friendly Streamlit dashboard.

## Features

- **Soil Health Assessment:** Classifies soil health as Poor, Moderate, or Good using soil parameters.
- **Soil Moisture Monitoring:** Reads soil moisture sensor data through an ESP32 microcontroller.
- **Soil Image Classification:** Identifies soil type from an uploaded image.
- **Crop Recommendation:** Recommends suitable crops based on available soil information and the identified soil type.
- **Nutrient Deficiency Analysis:** Evaluates Nitrogen, Phosphorus, and Potassium values for possible deficiencies.
- **Fertilizer Guidance:** Provides fertilizer-related suggestions based on available nutrient and crop information.

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- TensorFlow / Keras
- Pandas and NumPy
- Pillow (PIL)
- Joblib / Pickle
- PySerial
- Arduino IDE

## Hardware Used

- ESP32 development board
- Capacitive soil moisture sensor
- Female-to-female jumper wires
- USB cable
- Laptop or computer

## System Workflow

1. Enter soil parameters in the Streamlit application.
2. Assess soil health using the trained Machine Learning model.
3. Read soil moisture data from the ESP32 sensor.
4. Upload a soil image for soil type classification.
5. Generate crop recommendations based on the available soil information.
6. View nutrient deficiency analysis and fertilizer guidance.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/dhumaladiti03-sudo/AgroSense.git
cd AgroSense
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your web browser.

## ESP32 Soil Moisture Sensor Setup

Connect the capacitive soil moisture sensor to the ESP32:

| Sensor Pin | ESP32 Pin     |
| ---------- | ------------- |
| VCC        | 3.3V          |
| GND        | GND           |
| AO         | GPIO 34       |
| DO         | Not connected |

Connect the ESP32 to the computer using a USB cable. Make sure the serial port configured in `src/soil_moisture_sensor.py` matches the COM port assigned by your computer.

## Project Structure

```text
AgroSense/
├── app.py
├── requirements.txt
├── style.css
├── src/
│   ├── soil_moisture_sensor.py
│   ├── nutrient_analysis.py
│   ├── soil_image_prediction.py
│   └── combined_crop_prediction.py
├── models/
├── data/
└── assets/
```


## Future Scope

- Add more IoT sensors for additional soil parameters.
- Develop an app with multiregional languages.
- Extend crop recommendations other than indian crops.

## Authors

- Jiya Pashte
- Aditi Dhumal
- Tanvi Markad

## License

This project was developed for academic purposes.
