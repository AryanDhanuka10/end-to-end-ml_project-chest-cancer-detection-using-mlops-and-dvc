import os
import requests
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Model file path
MODEL_PATH = "artifacts/training/model.h5"
MODEL_URL = "https://github.com/AryanDhanuka10/end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc/releases/latest/download/model.h5"

# Function to download model if missing
def download_model():
    """Download the model file if it's missing."""
    if not os.path.exists(MODEL_PATH):
        st.warning("🔄 Model not found. Downloading from GitHub...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            st.success("✅ Model downloaded successfully!")
        else:
            st.error(f"❌ Failed to download model: {response.status_code}")
            return False
    return True

# Ensure the model is available
if download_model():
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")

# Function to preprocess the image
def preprocess_image(image):
    """Preprocesses an image for model prediction."""
    img = image.resize((224, 224))  # Resize image to model's expected input size
    img = img.convert("RGB")  # Ensure image has 3 channels
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to predict the class of an image
def predict_image(image):
    """Predicts the class of the given image using the model."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)

    st.write(f"🔍 Raw Predictions: {predictions}")  # Debugging Output

    if predictions.shape[1] != 4:  # Ensure correct shape
        st.error(f"❌ Unexpected prediction shape: {predictions.shape}")
        return "Error", 0.0

    class_names = [
        "Normal",
        "Adenocarcinoma",
        "Large Cell Carcinoma",
        "Squamous Cell Carcinoma"
    ]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100

    return predicted_class, confidence



# Streamlit UI
st.title("🩺 Chest Cancer Detection using Deep Learning")
st.write("Upload a chest X-ray image, and the AI model will predict whether it's Normal or Cancerous.")

uploaded_file = st.file_uploader("📤 Upload a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        predicted_class, confidence = predict_image(image)
        st.write(f"### 🏥 Prediction: **{predicted_class}**")
        st.write(f"### 🎯 Confidence: **{confidence:.2f}%**")
