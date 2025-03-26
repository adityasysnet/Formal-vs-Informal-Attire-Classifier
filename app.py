import streamlit as st
import torch
import clip
from PIL import Image
import cv2
import numpy as np
import os

# Ensure the static directory exists
os.makedirs("static/captured_images", exist_ok=True)
os.makedirs("static/uploaded_images", exist_ok=True)

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define class labels for attire classification
class_labels = ["A person in formal attire.", "A person in informal attire."]
text_inputs = clip.tokenize(class_labels).to(device)

def classify_attire(image):
    """
    Classifies an image as formal or informal attire using OpenAI CLIP.
    
    :param image: PIL Image object
    :return: Classification result and confidence scores
    """
    # Preprocess the image
    image = preprocess(image).unsqueeze(0).to(device)
    
    # Compute similarity between image and text prompts
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    # Extract results
    formal_score, informal_score = similarity[0].tolist()
    result = "Formal" if formal_score > informal_score else "Informal"

    return {
        "result": result,
        "confidence": {
            "Formal": round(formal_score * 100, 2),
            "Informal": round(informal_score * 100, 2)
        }
    }

# Streamlit UI
st.title("Attire Classification ")
st.write("Upload an image or capture from webcam to classify attire as Formal or Informal.")

# Upload image feature
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Live camera capture
capture_image = st.button("Capture Image from Webcam")
if capture_image:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_path = f"static/captured_images/captured_{len(os.listdir('static/captured_images')) + 1}.jpg"
        image.save(image_path)
        st.image(image, caption="Captured Image", use_column_width=True)
        result = classify_attire(image)
        st.write(f"**Classification:** {result['result']}")
        st.write(f"**Confidence Scores:**")
        st.write(f"Formal: {result['confidence']['Formal']}%")
        st.write(f"Informal: {result['confidence']['Informal']}%")
    else:
        st.error("Failed to capture image.")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_path = f"static/uploaded_images/uploaded_{len(os.listdir('static/uploaded_images')) + 1}.jpg"
    image.save(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    result = classify_attire(image)
    
    st.write(f"**Classification:** {result['result']}")
    st.write(f"**Confidence Scores:**")
    st.write(f"Formal: {result['confidence']['Formal']}%")
    st.write(f"Informal: {result['confidence']['Informal']}%")
