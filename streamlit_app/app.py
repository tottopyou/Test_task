import streamlit as st
import requests
from PIL import Image
import io

st.title("Task 2 - Road Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Road Detection"):
        # Send the uploaded image to the Flask app
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://task2_container:5002/process", files=files)

        if response.status_code == 200:
            # Display the processed image
            result_img = Image.open(io.BytesIO(response.content))
            st.image(result_img, caption="Road Detection Result", use_column_width=True)
        else:
            st.error("Road detection failed. Please try again.")
