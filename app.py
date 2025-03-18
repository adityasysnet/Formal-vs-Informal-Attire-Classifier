import streamlit as st
import torch
import clip
from PIL import Image
# from ultralytics import YOLO
# import numpy as np

# ---- Set up UI Layout (MUST BE FIRST STREAMLIT COMMAND) ----
st.set_page_config(
    page_title="Formal vs Informal Classifier",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Sidebar Navigation ----
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "📸 Classify Image", "ℹ️ About"])

# ---- Dark Mode Toggle ----
# theme = st.sidebar.radio("🎨 Theme:", ["🌞 Light Mode", "🌙 Dark Mode"])

# # Apply dark mode styles if selected
# if theme == "🌙 Dark Mode":
#     st.markdown(
#         """
#         <style>
#         body { background-color: #1E1E1E; color: #ffffff; }
#         .stTextInput>div>div>input { background-color: #333; color: white; }
#         .stButton>button { background-color: #666; color: white; border: 1px solid white; }
#         .stFileUploader>div>div>button { background-color: #666; color: white; }
#         .stSidebar>div { background-color: #333; color: white; }
#         .stSelectbox>div>div>div>div>input { background-color: #333; color: white; }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

# ---- Load CLIP Model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---- Footer Function ----
def add_footer():
    st.markdown("""
        <hr style="border:1px solid gray">
        <p style="text-align:center;">
            Made with ❤️ using Streamlit & OpenAI CLIP | 
            <a href="https://github.com/adityasysnet" target="_blank">GitHub</a>
        </p>
    """, unsafe_allow_html=True)

# ---- Page Handling ----
if page == "🏠 Home":
    st.title("👔 Formal vs Informal Attire Classifier")
    st.write("""
        Welcome to the **AI-Powered Attire Classifier**!  
        📸 Upload or capture an image, and the model will classify it as **Formal** or **Informal**.
        
        - Uses **OpenAI CLIP** for classification  
        - Works on **both desktop & mobile**  
        - Supports **live camera capture**  
        
        **🔍 Get Started:** Click on **"📸 Classify Image"** in the sidebar!
    """)

elif page == "📸 Classify Image":
    st.title("📸 Classify Image")
    st.write("Upload an image or take a picture, and I'll classify it!")

    # Choose input method
    option = st.radio("Choose Image Input Method:", ["📸 Camera", "📤 Upload Image"])

    # Image input handling
    uploaded_image = None
    IMAGE_SIZE = (500,500)
    if option == "📸 Camera":
        uploaded_image = st.camera_input("Take a picture")
    else:
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded/captured image
        image = Image.open(uploaded_image).convert("RGB")
        resized_image = image.resize(IMAGE_SIZE)
        st.image(resized_image, caption="📷 Input Image", use_column_width=True)

        # Preprocess the image
        image_input = preprocess(resized_image).unsqueeze(0).to(device)

        # Define text labels
        # text_labels = ["Formal attire", "Informal attire"]

        text_labels = ['Formal', 'Informal', 'Casual', 'Not a human']
#         text_labels = [
#     "Suit Formal",
#     "Dress Shirt Formal",
#     "Shirt Formal",
#     "Collar Shirt Formal",
#     "check shirt Formal",
#     "half-sleeve shirts Formal",
#     "Tie Formal",
#     "Bow Tie Formal",
#     "Dress Pants Formal",
#     "Blazer Formal",
#     "Waistcoat Formal",
#     "Dress Shoes Formal",
#     "Cufflinks Formal",
#     "Pocket Square Formal",
#     "Formal Belt Formal",
#     "Business Suit Formal",
#     "Formal Dress Formal",
#     "Cocktail Dress Formal",
#     "Gown Formal",
#     "Blouse Formal",
#     "Pencil Skirt Formal",
#     "High Heels Formal",
#     "Stockings Formal",
#     "Jewelry Formal",
#     "Handbag Formal",


#     "T-Shirts Informal",
#     "Jeans Informal",
#     "Hoodies Informal",
#     "Sweatpants Informal",
#     "Sneakers Informal",
#     "Cargo Pants Informal",
#     "Polo Shirts Informal",
#     "Flannel Shirts Informal",
#     "Denim Jackets Informal",
#     "Shorts Informal",
#     "Joggers Informal",
#     "Tank Tops Informal",
#     "Baseball Caps Informal",
#     "Graphic Tees Informal",
#     "Oversized Hoodies Informal",
#     "Leggings Informal",
#     "Yoga Pants Informal",
#     "Sportswear Informal",
#     "Tracksuits Informal",
#     "Gym Clothes Informal",

#     "NOT A HUMAN, cat, dog, computer, phone, car, tree", 
#     "house, table, chair, sofa, bed, window",
#     "door, book, cup, plate, fork, knife",
#     "spoon, bowl, banana, apple",
#     "sandwich, orange, broccoli, carrot",
#     "hot dog, pizza, donut, cake, chair",
#     "couch, potted plant, bed, dining table",
#     "toilet, TV, laptop, mouse, remote, keyboard",
#     "cell phone, microwave, oven, toaster, sink",
#     "refrigerator, blender, book, clock",
#     "vase, scissors, teddy bear",
#     "hair dryer, toothbrush, etc.",
# ]

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
        # Display confidence scores for both classes
        st.write("### 🔍 Prediction Breakdown:")
        for i, label in enumerate(text_labels):
            st.write(f"- **{label}:** {similarity[i].item() * 100:.2f}%")


elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.write("""
        This application classifies images into **Formal** or **Informal Attire** using OpenAI's **CLIP model**.
        
        **Features:**
        - 📸 Capture images from a camera
        - 📤 Upload an image
        - 🏆 AI-based classification with **CLIP**
        - 🌐 Mobile-friendly UI
        
        **Developed by:** Aditya Sharma  
        **GitHub:** [Click Here](https://github.com/adityasysnet)
    """)

    if st.button("🔄 Try Another Image"):
        st.rerun()  # NEW (Correct)


# ---- Footer ----
add_footer()
