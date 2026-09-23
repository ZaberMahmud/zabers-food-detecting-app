import streamlit as st
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
from collections import Counter
import pandas as pd
import numpy as np
import os

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Zabers Food Detecting App",
    page_icon="🍴",
    layout="centered"
)

st.title("🍴 ZABERS FOOD DETECTING APP")
st.info("🤖 AI Food Classification - YOLO11")

# ============================================
# MODEL
# ============================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "food_classifier.pt"
)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# ============================================
# 20 FOOD CLASSES
# ============================================

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

# ============================================
# SESSION MEMORY
# ============================================

if "detecting" not in st.session_state:
    st.session_state.detecting = False

if "captured_images" not in st.session_state:
    st.session_state.captured_images = []

if "predictions" not in st.session_state:
    st.session_state.predictions = []

# ============================================
# DETECTION CONTROL
# ============================================

st.divider()

st.subheader("🍽️ FOOD DETECTION CONTROL")

c1, c2 = st.columns(2)

with c1:
    if st.button(
        "▶️ START DETECTION",
        use_container_width=True
    ):
        st.session_state.detecting = True
        st.rerun()

with c2:
    if st.button(
        "⏹️ STOP DETECTION",
        use_container_width=True
    ):
        st.session_state.detecting = False
        st.rerun()

# ============================================
# CAMERA ON / OFF
# ============================================

st.subheader("📷 CAMERA ON/OFF")

camera_on = st.toggle(
    "Camera ON/OFF",
    value=st.session_state.detecting
)

# ============================================
# CAMERA DETECTION
# ============================================

if camera_on and st.session_state.detecting:

    st.success("🟢 AI FOOD DETECTION RUNNING")

    photo = st.camera_input(
        "CAPTURE FOOD IMAGE",
        label_visibility="collapsed"
    )

    if photo is not None:

        # ====================================
        # LOAD IMAGE
        # ====================================

        image = Image.open(photo).convert("RGB")

        # Convert PIL image to NumPy RGB array
        image_array = np.array(image)

        # ====================================
        # YOLO PREDICTION
        # ====================================

        results = model.predict(
            source=image_array,
            imgsz=224,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # ====================================
        # TOP-1 PREDICTION
        # ====================================

        predicted_index = result.probs.top1

        predicted_food = result.names[predicted_index]

        confidence = float(
            result.probs.top1conf
        )

        # ====================================
        # STORE DETECTION
        # ====================================

        st.session_state.captured_images.append(
            image
        )

        st.session_state.predictions.append({
            "food": predicted_food,
            "confidence": confidence,
            "time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        })

        # ====================================
        # SHOW RESULT
        # ====================================

        st.divider()

        st.subheader("🤖 AI DETECTION RESULT")

        st.image(
            image,
            caption="Captured Food",
            use_container_width=True
        )

        food_display = (
            predicted_food
            .replace("_", " ")
            .title()
        )

        st.success(
            f"🍴 Predicted Food: **{food_display}**"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.success(
            f"⚡ Detection #{len(st.session_state.predictions)} complete!"
        )

# ============================================
# DETECTION GALLERY
# ============================================

st.divider()

st.subheader(
    f"📸 DETECTION GALLERY - Total: "
    f"{len(st.session_state.captured_images)}"
)

if st.session_state.captured_images:

    cols = st.columns(3)

    recent_count = min(
        6,
        len(st.session_state.captured_images)
    )

    for i in range(recent_count):

        index = (
            len(st.session_state.captured_images)
            - recent_count
            + i
        )

        food_name = (
            st.session_state.predictions[index]["food"]
            .replace("_", " ")
            .title()
        )

        with cols[i % 3]:

            st.image(
                st.session_state.captured_images[index],
                caption=food_name,
                use_container_width=True
            )

# ============================================
# FOOD DETECTION REPORT
# ============================================

st.divider()

st.subheader("📊 FOOD DETECTION REPORT")

if st.session_state.predictions:

    food_counts = Counter(
        item["food"]
        for item in st.session_state.predictions
    )

    report_data = []

    for food, count in food_counts.items():

        report_data.append({
            "Food": food.replace(
                "_", " "
            ).title(),
            "Count": count
        })

    report_df = pd.DataFrame(
        report_data
    )

    st.table(report_df)

    st.metric(
        "Total Food Images Detected",
        len(st.session_state.predictions)
    )

else:

    st.info("No food has been detected yet.")

# ============================================
# DOWNLOAD REPORT
# ============================================

if st.session_state.predictions:

    report = "ZABERS FOOD DETECTION REPORT\n"
    report += "=" * 50 + "\n"
    report += f"Date: {datetime.now()}\n\n"

    report += "DETECTIONS\n"
    report += "-" * 50 + "\n"

    for i, item in enumerate(
        st.session_state.predictions,
        start=1
    ):

        food_name = (
            item["food"]
            .replace("_", " ")
            .title()
        )

        report += (
            f"{i}. {food_name} | "
            f"Confidence: "
            f"{item['confidence'] * 100:.2f}% | "
            f"{item['time']}\n"
        )

    report += "\nFOOD SUMMARY\n"
    report += "-" * 50 + "\n"

    for food, count in food_counts.items():

        food_name = (
            food
            .replace("_", " ")
            .title()
        )

        report += f"{food_name}: {count}\n"

    report += "\n"
    report += (
        f"Total Images: "
        f"{len(st.session_state.predictions)}\n"
    )

    st.download_button(
        "📥 DOWNLOAD FOOD DETECTION REPORT",
        report,
        file_name="zabers_food_detection_report.txt",
        use_container_width=True
    )

# ============================================
# STANDBY
# ============================================

elif not camera_on:

    st.warning(
        "🔴 Camera OFF - Detection Standby"
    )

else:

    st.info(
        "Press START DETECTION to begin."
    )
