import random

import streamlit as st
from PIL import Image
from ultralytics import YOLO
from utils import predict_captcha

cestat_model = YOLO("model/best_cestat.pt")

# List of random CESTAT images
cestat_images = ["images/captcha_image3001.png"]

def load_random_image(image_list):
    """Load a random image from a list of images."""
    return Image.open(random.choice(image_list))


def get_image_from_upload():
    """Get the uploaded image from Streamlit file uploader."""
    uploaded_file = st.file_uploader(
        "Choose a CESTAT image...", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None:
        return uploaded_file, Image.open(uploaded_file)
    return None, None


# Streamlit app layout
st.title("CESTAT Captcha Prediction App")

# Add radio buttons for image input method
input_choice = st.radio(
    "Choose input method", ("Upload Image", "Use Random CESTAT Image")
)

# Image variable initialization
image = None
uploaded_file = None

# Load image based on input method
if input_choice == "Upload Image":
    uploaded_file, image = get_image_from_upload()
    if image is not None:
        st.image(image, caption="Uploaded CESTAT Image", use_container_width=True)

elif input_choice == "Use Random CESTAT Image":
    # Load a random CESTAT image
    image = load_random_image(cestat_images)
    st.image(image, caption="Random CESTAT Image", use_container_width=True)

# If an image is available, predict the captcha
if image is not None:
    with st.spinner("Processing....."):
        if st.button("Predict Captcha"):
            captcha_text = predict_captcha(image, cestat_model)
            st.success(f"Predicted CESTAT Captcha: {captcha_text}")
