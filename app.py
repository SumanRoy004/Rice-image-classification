import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
tf.get_logger().setLevel('ERROR')         # Suppress Keras warnings

# Load the pre-trained CNN model
@st.cache_resource
def load_cnn_model():
    model = load_model('Rice_image_classification_cnn_model.h5')
    return model

cnn = load_cnn_model()

# Class names
class_names = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}

# Function to preprocess and predict
def predict_rice_variety(image):
    image = image.resize((160, 160))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = cnn.predict(image_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    return class_names[predicted_class], confidence, prediction[0]

# --- Streamlit UI ---
st.title("🌾 Rice Variety Classification")
st.markdown("Upload an image of rice grains to classify its variety. The app supports the following rice varieties:")
st.markdown("""
- **Arborio**
- **Basmati**
- **Ipsala**
- **Jasmine**
- **Karacadag**
""")

# Helpful Tips
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    - Upload a clear image of rice grains.
    - Supported formats: JPG, JPEG, PNG.
    - Click **Classify** to see the predicted variety and class probabilities.
    - Or click on a **sample image** to see how the model responds.
    """)

# Show example images
st.subheader("📸 Sample Images")
sample_folder = "sample_images"

cols = st.columns(len(class_names))
for i, (label_id, label_name) in enumerate(class_names.items()):
    image_path = os.path.join(sample_folder, f"{label_name}.jpg")
    if os.path.exists(image_path):
        with cols[i]:
            st.image(image_path, caption=label_name,use_container_width=True)
            if st.button(f"Try {label_name}"):
                sample_image = Image.open(image_path)
                pred, conf, proba = predict_rice_variety(sample_image)
                st.success(f"Prediction: {pred}")
                st.info(f"Confidence: {conf:.2%}")
                st.subheader("Class Probabilities:")
                for j, prob in enumerate(proba):
                    st.write(f"{class_names[j]}: {prob:.2%}")

# File uploader
st.subheader("📤 Upload Your Own Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Classify"):
        with st.spinner('🔍 Classifying...'):
            prediction, confidence, probabilities = predict_rice_variety(image)
        
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2%}")
        
        st.subheader("Class Probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_names[i]}: {prob:.2%}")
