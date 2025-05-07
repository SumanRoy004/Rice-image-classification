import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs (0-3)
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
    # Resize image to 160x160 as per model's expected input
    image = image.resize((160, 160))
    # Convert to array and normalize
    image_array = img_to_array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # Get prediction
    prediction = cnn.predict(image_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    return class_names[predicted_class], confidence

# Streamlit UI
st.title("Rice Variety Classification")
st.write("Upload an image of rice grains to classify its variety.The app supports the following rice grains:")
st.write("""
        - Arborio
        - Basmati
        - Ipsala
        - Jasmine
        - Karacadag
        """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Make prediction when button is clicked
    if st.button("Classify"):
        with st.spinner('Classifying...'):
            prediction, confidence = predict_rice_variety(image)
        
        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2%}")
        
        # Show class probabilities
        st.subheader("Class Probabilities:")
        proba = cnn.predict(img_to_array(image.resize((160, 160))).reshape(1, 160, 160, 3)/255.0)[0]
        for i, prob in enumerate(proba):
            st.write(f"{class_names[i]}: {prob:.2%}")