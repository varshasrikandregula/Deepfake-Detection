import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🎭",
    layout="wide"
)

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_my_model():
    return load_model("best_xception_improved.keras")

model = load_my_model()

THRESHOLD = 0.35

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("🎭 DeepFake Detection")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Image Detection",
        "Video Detection",
        "About"
    ]
)

# ===================================================
# IMAGE DETECTION
# ===================================================
if menu == "Image Detection":

    st.title("🖼️ Image DeepFake Detection")

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([0.8, 1.2])

        # ------------------------
        # LEFT SIDE
        # ------------------------
        with col1:

            st.subheader("🖼️ Image Preview")

            st.image(
                img,
                width=250
            )

        # ------------------------
        # RIGHT SIDE
        # ------------------------
        with col2:

            st.subheader("🔍 Detection")

            if st.button(
                "Analyze Image",
                use_container_width=True
            ):

                with st.spinner("Analyzing Image..."):

                    img_resized = img.resize((224, 224))

                    img_array = image.img_to_array(
                        img_resized
                    )

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    img_array = preprocess_input(
                        img_array
                    )

                    pred = model.predict(
                        img_array,
                        verbose=0
                    )[0][0]

                if pred >= THRESHOLD:

                    confidence = float(pred * 100)

                    st.success(
                        "✅ REAL IMAGE DETECTED"
                    )

                else:

                    confidence = float(
                        (1 - pred) * 100
                    )

                    st.error(
                        "🚨 FAKE IMAGE DETECTED"
                    )

                st.markdown(
                    "### Prediction Confidence"
                )

                st.markdown(
                    f"# {confidence:.2f}%"
                )

                st.progress(
                    float(confidence) / 100
                )

                st.metric(
                    "Raw Prediction",
                    f"{float(pred):.4f}"
                )
# ===================================================
# VIDEO DETECTION
# ===================================================
elif menu == "Video Detection":

    st.title("🎥 Video DeepFake Detection")
    st.write("Upload a video to determine whether it is REAL or FAKE.")

    uploaded_video = st.file_uploader(
        "Choose a Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(uploaded_video.read())
            video_path = tmp.name

        col1, col2 = st.columns([0.45, 0.9])

        # ------------------------
        # LEFT SIDE
        # ------------------------
        with col1:

            st.subheader("🎬 Video Preview")

            # Thumbnail Removed
            st.video(video_path)

        # ------------------------
        # RIGHT SIDE
        # ------------------------
        with col2:

            st.subheader("🔍 Detection")

            if st.button(
                "Analyze Video",
                use_container_width=True
            ):

                predictions = []

                cap = cv2.VideoCapture(video_path)

                frame_count = 0

                with st.spinner(
                    "Analyzing Video..."
                ):

                    while True:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        if frame_count % 30 == 0:

                            frame = cv2.cvtColor(
                                frame,
                                cv2.COLOR_BGR2RGB
                            )

                            img = Image.fromarray(frame)
                            img = img.resize((224, 224))

                            img_array = image.img_to_array(img)
                            img_array = np.expand_dims(
                                img_array,
                                axis=0
                            )

                            img_array = preprocess_input(
                                img_array
                            )

                            pred = model.predict(
                                img_array,
                                verbose=0
                            )[0][0]

                            predictions.append(pred)

                        frame_count += 1

                cap.release()

                if len(predictions) == 0:
                    st.error("No frames could be analyzed.")
                else:

                    avg_pred = float(np.mean(predictions))

                    st.write(
                        "Frames Analyzed:",
                        len(predictions)
                    )

                    st.write(
                        "Average Prediction:",
                        avg_pred
                    )

                    THRESHOLD = 0.35

                    if avg_pred >= THRESHOLD:

                        confidence = float(
                            avg_pred * 100
                        )

                        st.success(
                            "✅ REAL VIDEO DETECTED"
                        )

                    else:

                        confidence = float(
                            (1 - avg_pred) * 100
                        )

                        st.error(
                            "🚨 FAKE VIDEO DETECTED"
                        )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.progress(
                        float(confidence) / 100
                    )
# ===================================================
# ABOUT
# ===================================================
else:

    st.title("📘 About Project")

    st.subheader(
        "DeepFake Detection Using XceptionNet"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "89.29%"
        )

    with col2:
        st.metric(
            "Precision",
            "89.00%"
        )

    with col3:
        st.metric(
            "Recall",
            "89.00%"
        )

    with col4:
        st.metric(
            "AUC",
            "96.31%"
        )

    st.markdown("---")

    st.info("""
Model : XceptionNet

Dataset : Celeb-DF

Input Size : 224 × 224

Validation Accuracy : 89.97%

Test Accuracy : 89.29%

Test AUC : 0.9631

Technology Stack :
• TensorFlow
• Keras
• OpenCV
• Streamlit
• Python

Features :
• Image DeepFake Detection
• Video DeepFake Detection
• Confidence Score
""")