import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pyttsx3

# ------------------- CONFIG -------------------
MODEL_PATH = "asl_model.keras"           # or "asl_model.h5"
CLASS_NAMES_PATH = "class_names.npy"
IMG_HEIGHT, IMG_WIDTH = 64, 64

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    return np.load(CLASS_NAMES_PATH)

model = load_model()
class_names = load_class_names()

# ------------------- SPEAK FUNCTION -------------------
def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Change to voices[1].id for female
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

# ------------------- PREDICTION FUNCTION -------------------
def predict(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    prediction = model.predict(img_array)
    index = np.argmax(prediction[0])
    label = class_names[index]
    confidence = np.max(prediction[0])
    return label, confidence

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="ASL Sign Classifier", page_icon="🤟", layout="centered")

# ---------- CSS Styling ----------
st.markdown("""
    <style>
        .main-title {
            font-size: 2.8em;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .description {
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 20px;
            color: #555;
        }
        .prediction-box {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-size: 1.2rem;
            color: #000;
        }
        .stButton > button {
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="main-title">🤟 ASL Sign Language Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload an image of an ASL hand sign and get an instant spoken prediction!</div>', unsafe_allow_html=True)

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📁 Upload ASL Image", type=["jpg", "png", "jpeg"])

# ---------- Prediction Flow ----------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=False, width=250)

    if st.button("🎯 Predict Sign"):
        with st.spinner("Analyzing the sign..."):
            label, confidence = predict(image)

            # Handle special signs
            spoken_label = str(label)
            if spoken_label == "SPACE":
                spoken_label = "space"
            elif spoken_label == "DELETE":
                spoken_label = "delete"
            elif spoken_label == "NOTHING":
                spoken_label = "nothing"

            # Save prediction
            st.session_state['prediction'] = {
                'label': label,
                'confidence': confidence
            }

            # Automatically speak
            speak(spoken_label)

# ---------- Show Result ----------
if 'prediction' in st.session_state:
    label = st.session_state['prediction']['label']
    confidence = st.session_state['prediction']['confidence']

    st.markdown(f"""
    <div class="prediction-box">
        ✅ <strong>Predicted Sign:</strong>
        <span style="font-size: 1.5rem; font-weight: bold;">{label}</span><br><br>
        🔍 <strong>Confidence:</strong> {confidence*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # Optional: Repeat audio
    if st.button("🔊 Repeat Prediction"):
        speak(str(label))
