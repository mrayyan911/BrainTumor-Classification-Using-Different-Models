import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Title
st.markdown("<h1 style='text-align:center; color:#1F77B4;'>🧠 Brain Tumor Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6c757d;'>Upload a brain MRI image and select the model to classify tumor type.</p>", unsafe_allow_html=True)

# Model options and their paths (update paths if needed)
MODEL_INFO = {
    "CNN": {
        "path": "CNNforBrainTumor.h5",
        "accuracy": 0.93
    },
    "VGG16": {
        "path": "VGG16forBrainTumor.h5",
        "accuracy": 0.93
    }
}

class_labels = ['Glioma', 'Meningioma', 'Pituitary Tumor', 'No Tumor']

# Sidebar: Model selector
model_name = st.sidebar.selectbox("Select Model", list(MODEL_INFO.keys()))

# Load model (cache to avoid reloading each time)
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        return model
    else:
        return None

model = load_model(MODEL_INFO[model_name]["path"])

if model is None:
    st.error(f"Model file not found: {MODEL_INFO[model_name]['path']}. Please place it in the app directory.")
    st.stop()

# Upload image with drag & drop
uploaded_file = st.file_uploader("Drag & drop brain MRI image (jpg/png)", type=["jpg", "jpeg", "png"])

def preprocess_image(img: Image.Image, target_size=(224,224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dim
    return img_array

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        # Preprocess image for prediction
        input_arr = preprocess_image(img)

        # Predict
        preds = model.predict(input_arr)
        pred_idx = np.argmax(preds)
        confidence = preds[0][pred_idx]

        # Show results
        st.markdown(f"### Prediction: **{class_labels[pred_idx]}**")
        st.markdown(f"### Confidence: **{confidence*100:.2f}%**")
        st.markdown(f"### Model Accuracy: **{MODEL_INFO[model_name]['accuracy']*100:.2f}%**")

else:
    st.info("Please upload a brain MRI image for prediction.")

st.markdown("<hr>")
st.markdown("<p style='text-align:center; color:#a9a9a9;'>Developed by Rayyan and Khizer</p>", unsafe_allow_html=True)
