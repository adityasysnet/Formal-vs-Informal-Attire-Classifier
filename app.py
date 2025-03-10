import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise use CPU
model, preprocess = clip.load("ViT-B/32", device=device)

# Streamlit UI
st.title("👔 Formal vs Informal Attire Classifier")
st.write("Upload an image or take a picture, and I'll classify it as **Formal** or **Informal**!")

# Choose input method
option = st.radio("Choose Image Input Method:", ["📸 Camera", "📤 Upload Image"])

# Image input handling
uploaded_image = None

if option == "📸 Camera":
    uploaded_image = st.camera_input("Take a picture")
else:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded/captured image
    image = Image.open(uploaded_image)
    st.image(image, caption="📷 Input Image", use_column_width=True)

    # Convert image to RGB and preprocess for CLIP
    image = image.convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Define text labels
    text_labels = ["Formal attire", "Informal attire"]
    text_inputs = clip.tokenize(text_labels).to(device)

    # Perform classification
    with st.spinner("🌀 Analyzing image..."):
        with torch.no_grad():
            # Get image and text features
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

            # Compute similarity between image and text
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)

            # Get the predicted label
            predicted_label = text_labels[similarity.argmax().item()]

    # Display the result
    st.success(f"### 🏆 Prediction: **{predicted_label}**")

    # Add a "Try Another Image" button to clear the uploaded image
    if st.button("🔄 Try Another Image"):
        st.experimental_rerun()  # Refreshes the app
