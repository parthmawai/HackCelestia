import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# Load pre-trained models (make sure these models are available)
# Load a face recognition model (for person similarity)
# Here, we'll assume you have a model trained for both tasks.

# Gender classification model (using a placeholder model)
gender_model = tf.keras.models.load_model("path_to_gender_model")  # Replace with your gender model
# Similarity detection model (replace with your actual model)
face_recognition_model = tf.keras.models.load_model("path_to_face_recognition_model")

# Preprocessing functions
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def compare_faces(image1, image2):
    # For simplicity, assume the face_recognition_model outputs a similarity score
    image1_embedding = face_recognition_model.predict(preprocess_image(image1))
    image2_embedding = face_recognition_model.predict(preprocess_image(image2))
    
    # Calculate similarity (e.g., using cosine similarity or Euclidean distance)
    similarity_score = np.linalg.norm(image1_embedding - image2_embedding)
    return similarity_score

def predict_gender(image):
    image_preprocessed = preprocess_image(image)
    prediction = gender_model.predict(image_preprocessed)
    gender = np.argmax(prediction, axis=1)
    return "Male" if gender == 0 else "Female"

# Streamlit UI
st.title("Image Processing with ML Models")

# Image similarity section
st.header("Check if Two People Are the Same")
uploaded_image1 = st.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
uploaded_image2 = st.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"])

if uploaded_image1 and uploaded_image2:
    image1 = Image.open(uploaded_image1)
    image2 = Image.open(uploaded_image2)
    
    similarity_score = compare_faces(image1, image2)
    st.write(f"Similarity score between the two images: {similarity_score}")
    if similarity_score < 0.5:  # Threshold for similarity
        st.write("The person in the images are likely the same.")
    else:
        st.write("The person in the images are likely different.")

# Gender classification section
st.header("Predict Gender from Image")
uploaded_image = st.file_uploader("Upload an image to predict gender", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    gender = predict_gender(image)
    st.write(f"The predicted gender is: {gender}")