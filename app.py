
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import os
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Zabers Food Detecting App",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Zaber's Food Detecting App")
st.write("Upload a food image or take a photo to identify the food.")

# --------------------------------------------------
# FOOD CLASSES
# --------------------------------------------------

FOOD_CLASSES = [
    "pizza",
    "hamburger",
    "chicken_wings",
    "french_fries",
    "hot_dog",
    "tacos",
    "sushi",
    "ramen",
    "fried_rice",
    "steak",
    "pancakes",
    "waffles",
    "donuts",
    "ice_cream",
    "chocolate_cake",
    "cheesecake",
    "caesar_salad",
    "lasagna",
    "spaghetti_bolognese",
    "samosa"
]

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "food_classifier.pt"
)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_food(image):

    image = image.convert("RGB")

    image_array = np.array(image)

    results = model.predict(
        source=image_array,
        imgsz=224,
        device="cpu",
        verbose=False
    )

    result = results[0]

    predicted_index = result.probs.top1

    predicted_food = result.names[predicted_index]

    confidence = float(result.probs.top1conf)

    return image, predicted_food, confidence


# --------------------------------------------------
# INPUT METHOD
# --------------------------------------------------

st.subheader("Choose Image Source")

input_method = st.radio(
    "Select an option:",
    ["📷 Take Photo", "🖼️ Upload Image"],
    horizontal=True
)

image = None

# --------------------------------------------------
# CAMERA INPUT
# --------------------------------------------------

if input_method == "📷 Take Photo":

    camera_image = st.camera_input(
        "Take a photo of your food"
    )

    if camera_image is not None:
        image = Image.open(camera_image)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

else:

    uploaded_image = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if image is not None:

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    if st.button(
        "🔍 Detect Food",
        use_container_width=True
    ):

        with st.spinner("Analyzing food..."):

            processed_image, predicted_food, confidence = predict_food(image)

        # Format food name
        display_name = predicted_food.replace("_", " ").title()

        confidence_percent = confidence * 100

        st.success(
            f"🍽️ Prediction: **{display_name}**"
        )

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )

        # --------------------------------------------------
        # SAVE RESULT TO HISTORY
        # --------------------------------------------------

        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Food": display_name,
            "Confidence": f"{confidence_percent:.2f}%"
        })


# --------------------------------------------------
# HISTORY / GALLERY
# --------------------------------------------------

if st.session_state.history:

    st.divider()

    st.subheader("📋 Detection History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Download report

    csv_data = history_df.to_csv(index=False)

    st.download_button(
        label="⬇️ Download Detection Report",
        data=csv_data,
        file_name="zabers_food_detection_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🍽️ About")

    st.write(
        "Zaber's Food Detecting App uses a YOLO11s "
        "classification model to recognize 20 different foods."
    )

    st.write("### Supported Foods")

    for food in FOOD_CLASSES:
        st.write(
            f"• {food.replace('_', ' ').title()}"
        )

    st.divider()

    st.write("### Model")

    st.write("YOLO11s Classification")

    st.write("### Test Accuracy")

    st.write("88.6% Top-1")

    st.write("97.1% Top-5")
```
