import streamlit as st
import requests
import os
import cv2
import pandas as pd
import shutil
from PIL import Image
import io

UPLOAD_FOLDER_TASK1 = "images"
os.makedirs(UPLOAD_FOLDER_TASK1, exist_ok=True)  # Ensure directory exists

# Step 1: Template Matching
st.header("Task 1: Template Matching")

# Upload images
small_image_file = st.file_uploader("Upload Small Image (Template)", type=["jpg", "png"], key="small_image")
large_image_file = st.file_uploader("Upload Large Image (Search Area)", type=["jpg", "png"], key="large_image")

if small_image_file and large_image_file:
    # Save files locally
    small_image_path = os.path.join(UPLOAD_FOLDER_TASK1, small_image_file.name)
    large_image_path = os.path.join(UPLOAD_FOLDER_TASK1, large_image_file.name)

    with open(small_image_path, "wb") as f:
        f.write(small_image_file.getbuffer())
    with open(large_image_path, "wb") as f:
        f.write(large_image_file.getbuffer())

    st.image([small_image_path, large_image_path], caption=["Small Image", "Large Image"], use_container_width=True)

    # Template Matching
    small_image = cv2.imread(small_image_path, cv2.IMREAD_GRAYSCALE)
    large_image = cv2.imread(large_image_path, cv2.IMREAD_GRAYSCALE)

    # Scale Selection Mode
    scale_mode = st.radio("Select Scale Mode", options=["Use Default Scales", "Custom Scales"])

    if scale_mode == "Use Default Scales":
        scales = [0.15, 0.20, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    else:
        custom_scales = st.text_input("Enter custom scales (comma-separated, e.g., 0.1, 0.2, 0.5)", value="0.15,0.20,0.25")
        try:
            scales = [float(s.strip()) for s in custom_scales.split(",")]
        except ValueError:
            st.error("Invalid input for custom scales. Please enter valid numeric values separated by commas.")
            scales = []

    if scales:
        # Matching variables
        threshold = 0
        best_value = -1
        best_match = None
        best_scale = None
        best_top_left = None
        results = []  # Store results for each scale

        for scale in scales:
            resized_small_image = cv2.resize(small_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            result = cv2.matchTemplate(large_image, resized_small_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            results.append((scale, max_val, max_loc))
            if max_val > best_value:
                best_value = max_val
                best_match = resized_small_image
                best_scale = scale
                best_top_left = max_loc

        # Display all tested scales
        st.write("### Tested Scales and Correlation Values")
        results_df = pd.DataFrame(results, columns=["Scale", "Correlation", "Top Left"])
        st.dataframe(results_df)

        # Best match visualization
        if best_value >= threshold:
            st.success(f"Best match found with correlation {best_value:.2f} at scale {best_scale}")
            h, w = best_match.shape
            top_left = best_top_left
            bottom_right = (top_left[0] + w, top_left[1] + h)

            large_image_color = cv2.imread(large_image_path)
            detected_image = cv2.rectangle(large_image_color, top_left, bottom_right, (0, 255, 0), 3)
            result_path = os.path.join(UPLOAD_FOLDER_TASK1, "result.jpg")
            cv2.imwrite(result_path, large_image_color[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
            st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption="Detected Region", use_container_width=True)

            st.write(f"Top-left corner coordinates of the smaller image in the larger image: {top_left}")
            st.write(f"Result saved for further processing: {result_path}")
        else:
            st.error("No match found. Adjust the threshold or check the images.")

# Step 2: Road Detection
st.header("Task 2: Road Detection")

# Define base folder for RoadNet
BASE_FOLDER = "/app/DeepSegmentor/datasets/RoadNet"
SUBFOLDERS = ["test_image", "test_segment", "test_edge", "test_centerline"]
RESULT_IMAGES_FOLDER = "/app/DeepSegmentor/results/images"

# Check if GPU is available
gpu_available = False
if "gpu_checked" not in st.session_state:
    st.session_state.gpu_checked = False

if st.button("Check GPU Availability") or not st.session_state.gpu_checked:
    st.session_state.gpu_checked = True
    try:
        gpu_response = requests.get("http://task2_container:5002/gpu_availability")
        if gpu_response.status_code == 200 and gpu_response.json().get("gpu_available", False):
            gpu_available = True
            st.session_state.gpu_available = True
            st.write(f"GPU is available.")
        else:
            st.session_state.gpu_available = False
            st.warning("GPU is not available.")
    except Exception as e:
        st.error(f"Failed to check GPU availability: {e}")
        st.session_state.gpu_available = False

gpu_available = st.session_state.gpu_available

# Persist GPU/CPU selection
if "selected_device" not in st.session_state:
    st.session_state.selected_device = "GPU (if available)" if gpu_available else "CPU"

selected_device = st.radio(
    "Select device for computation:",
    options=["GPU (if available)", "CPU"],
    index=0 if st.session_state.selected_device == "GPU (if available)" else 1,
)

st.session_state.selected_device = selected_device
gpu_id = 0 if selected_device == "GPU (if available)" and gpu_available else -1

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
scaled_image = None  # Placeholder for the final image to save

if uploaded_file is not None:
    # Convert to PNG if the uploaded file is a JPG
    uploaded_file_name = uploaded_file.name
    if uploaded_file_name.lower().endswith(".jpg"):
        try:
            # Convert JPG to PNG
            image = Image.open(uploaded_file).convert("RGB")
            uploaded_file_name = os.path.splitext(uploaded_file_name)[0] + ".png"
            st.write(f"Converted JPG to PNG: {uploaded_file_name}")
        except Exception as e:
            st.error(f"Failed to convert JPG to PNG: {e}")
            uploaded_file = None
    else:
        image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Image scaling options
    scale_factor = st.radio(
        "Choose scaling for the image:",
        options=["Original", "x2", "x4", "x8"],
        index=0,
    )

    # Apply scaling if necessary
    try:
        if scale_factor != "Original":
            scale_multiplier = int(scale_factor[1])  # Extract the scaling multiplier (2, 4, or 8)
            scaled_width, scaled_height = image.size[0] * scale_multiplier, image.size[1] * scale_multiplier
            scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            st.write(f"Image scaled to: {scaled_width}x{scaled_height}")
        else:
            scaled_image = image
    except Exception as e:
        st.error(f"Failed to scale image: {e}")

    # Save the scaled image only when "Run Road Detection" is clicked
    if st.button("Run Road Detection"):
        if scaled_image is not None:
            # Save scaled image to each subfolder
            for subfolder in SUBFOLDERS:
                subfolder_path = os.path.join(BASE_FOLDER, subfolder)
                host_path = os.path.join(subfolder_path, uploaded_file_name)
                flask_path = f"/app/DeepSegmentor/datasets/RoadNet/{subfolder}/{uploaded_file_name}"

                try:
                    # Save the scaled image
                    with open(host_path, "wb") as f:
                        scaled_image.save(f, format="PNG")
                except Exception as e:
                    st.error(f"Failed to save file to {subfolder}: {e}")

            # Use the "test_image" folder path as the reference for Flask processing
            flask_path = f"/app/DeepSegmentor/datasets/RoadNet/test_image/{uploaded_file_name}"
            payload = {
                "file_path": flask_path,
                "gpu_id": gpu_id,
                "width": scaled_image.size[0],
                "height": scaled_image.size[1],
            }

            # Send request to Flask app
            response = requests.post("http://task2_container:5002/process", json=payload)

            # Handle response
            if response.status_code == 200:
                st.success("Road detection successful!")

                st.markdown("### Road Detection Predicted Image")
                try:
                    # Construct the expected predicted image filename
                    base_uploaded_name = os.path.splitext(uploaded_file_name)[0]
                    predicted_images = [file for file in os.listdir(RESULT_IMAGES_FOLDER) if file.endswith("_pred.png")]

                    # Look for the matching `_pred.png` file
                    matched_file = None
                    for file in predicted_images:
                        if base_uploaded_name in file:
                            matched_file = file
                            break

                    if matched_file:
                        predicted_image_path = os.path.join(RESULT_IMAGES_FOLDER, matched_file)
                        with open(predicted_image_path, "rb") as f:
                            result_image = Image.open(f)
                            st.image(result_image, caption=f"Predicted Result: {matched_file}", use_container_width =True)
                    else:
                        st.warning(f"No predicted result image found for {uploaded_file_name} in the results folder.")
                except Exception as e:
                    st.error(f"Failed to load the predicted result image: {e}")
            else:
                error_message = response.json().get("error", "Unknown error occurred.")
                st.error(f"Error: {error_message}")
