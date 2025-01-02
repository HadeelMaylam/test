import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import sqlite3
from deepface import DeepFace
import cv2
import numpy as np
import os
import shutil
from PIL import Image
import base64
import io

# Import existing functions
from FaceRecognition import (
    init_database,
    register_face,
    verify_face,
    get_all_users,
)

# Initialize the database
init_database()

class VideoCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

# Helper function to save image from webcam
def save_captured_image(video_transformer, filename="captured_image.jpg"):
    if video_transformer.frame is not None:
        cv2.imwrite(filename, video_transformer.frame)
        return filename
    return None

# Helper function to save uploaded image
def save_uploaded_image(uploaded_file):
    temp_path = f"temp_{os.urandom(4).hex()}.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_path

# Streamlit App
def main():
    st.title("Face Recognition System")

    # Sidebar menu
    menu = ["Home", "Register Face", "Verify Face", "View Registered Users"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the Face Recognition System! Use the sidebar to navigate.")

    elif choice == "Register Face":
        st.header("Register a New Face")
        name = st.text_input("Enter Person's Name")

        capture_mode = st.radio("Choose Image Input Method", ("Upload Image", "Capture from Webcam"))

        # Webcam setup
        captured_image_path = None
        if capture_mode == "Capture from Webcam":
            webrtc_ctx = webrtc_streamer(
                key="capture",
                video_transformer_factory=VideoCaptureTransformer,
            )
            if st.button("Capture Image"):
                video_transformer = webrtc_ctx.video_transformer
                if video_transformer is not None:
                    captured_image_path = save_captured_image(video_transformer)

        uploaded_file = None
        if capture_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        if st.button("Register"):
            if name:
                if uploaded_file:
                    image_path = save_uploaded_image(uploaded_file)
                elif captured_image_path:
                    image_path = captured_image_path
                else:
                    st.warning("Please capture or upload an image.")
                    return

                success, message = register_face(image_path, name)
                st.success(message) if success else st.error(message)

                # Clean up temporary files
                if captured_image_path and os.path.exists(captured_image_path):
                    os.remove(captured_image_path)
                if uploaded_file and os.path.exists(image_path):
                    os.remove(image_path)
            else:
                st.warning("Please enter the person's name.")

    elif choice == "Verify Face":
        st.header("Verify a Face")

        capture_mode = st.radio("Choose Image Input Method", ("Upload Image", "Capture from Webcam"))

        # Webcam setup
        captured_image_path = None
        if capture_mode == "Capture from Webcam":
            webrtc_ctx = webrtc_streamer(
                key="verify",
                video_transformer_factory=VideoCaptureTransformer,
            )
            if st.button("Capture Image"):
                video_transformer = webrtc_ctx.video_transformer
                if video_transformer is not None:
                    captured_image_path = save_captured_image(video_transformer)

        uploaded_file = None
        if capture_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        if st.button("Verify"):
            if uploaded_file:
                image_path = save_uploaded_image(uploaded_file)
            elif captured_image_path:
                image_path = captured_image_path
            else:
                st.warning("Please capture or upload an image.")
                return

            success, message = verify_face(image_path)
            st.success(message) if success else st.error(message)

            # Clean up temporary files
            if captured_image_path and os.path.exists(captured_image_path):
                os.remove(captured_image_path)
            if uploaded_file and os.path.exists(image_path):
                os.remove(image_path)

    elif choice == "View Registered Users":
        st.header("Registered Users")
        users = get_all_users()
        if users:
            for user in users:
                st.subheader(f"ID: {user[0]} - Name: {user[1]}")
                if user[2]:
                    image_data = base64.b64decode(user[2])
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, caption=user[1], width=200)
        else:
            st.info("No users registered yet.")

if __name__ == "__main__":
    main()
