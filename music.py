import os
import time
import webbrowser
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from keras.models import load_model

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)


MODEL_FACE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "mediapipe_models"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

FACE_LANDMARK_COUNT = 468
HAND_LANDMARK_COUNT = 21
FEATURES_FACE = FACE_LANDMARK_COUNT * 2  # x,y for each
FEATURES_HAND = HAND_LANDMARK_COUNT * 2  # x,y for each
FEATURES_TOTAL = FEATURES_FACE + (FEATURES_HAND * 2)  # face + 2 hands
DETECTION_WINDOW_SECONDS = 10
MIN_VALID_PREDICTIONS = 8
MIN_MAJORITY_RATIO = 0.4


def _ensure_mediapipe_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_MODEL_PATH.exists():
        urlretrieve(MODEL_FACE_URL, FACE_MODEL_PATH.as_posix())
    if not HAND_MODEL_PATH.exists():
        urlretrieve(MODEL_HAND_URL, HAND_MODEL_PATH.as_posix())


@st.cache_resource
def _load_landmarkers():
    _ensure_mediapipe_models()

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH.as_posix()),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH.as_posix()),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
    )

    face_landmarker = FaceLandmarker.create_from_options(face_options)
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    return face_landmarker, hand_landmarker


def _face_to_features(face_landmarks) -> list[float]:
    # Match the original feature engineering: offset all face landmarks
    # relative to landmark index 1.
    # Newer Mediapipe face models may return 478 landmarks (with iris points).
    # The trained model expects the original 468-landmark layout.
    face_landmarks = face_landmarks[:FACE_LANDMARK_COUNT]
    origin = face_landmarks[1]
    feats: list[float] = []
    for lm in face_landmarks:
        feats.append(lm.x - origin.x)
        feats.append(lm.y - origin.y)
    return feats


def _hand_to_features(hand_landmarks) -> list[float]:
    # Match the original feature engineering: offset all hand landmarks
    # relative to landmark index 8.
    origin = hand_landmarks[8]
    feats: list[float] = []
    for lm in hand_landmarks:
        feats.append(lm.x - origin.x)
        feats.append(lm.y - origin.y)
    return feats


# Load the trained model and labels (do it once per Streamlit run)
model = load_model("model.h5")
label = np.load("labels.npy")

st.header("🎭 Emotion-Based Music Recommender 🎵")

lang = st.text_input("🎤 Enter Language:")
singer = st.text_input("🎼 Enter Singer Name:")

if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

if lang and singer and not st.session_state["emotion"]:
    st.subheader("📷 Camera is ON. Detecting Emotion...")
    st.caption("Hold a steady expression for a few seconds for better accuracy.")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("⚠ Camera not detected! Please check your webcam.")
        st.stop()

    with st.spinner("Loading Mediapipe models (first time may take a while)..."):
        face_landmarker, hand_landmarker = _load_landmarkers()

    start_time = time.time()
    # Mediapipe's VIDEO-mode tasks require monotonically increasing timestamps.
    # Streamlit can re-run the script while caching the task objects, so we keep
    # the last used timestamp in session state across runs.
    last_ts_ms = int(st.session_state.get("mp_last_ts_ms", 0))
    predicted_labels: list[str] = []

    while time.time() - start_time < DETECTION_WINDOW_SECONDS:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠ Unable to read from camera!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.perf_counter() * 1000)
        if ts_ms <= last_ts_ms:
            ts_ms = last_ts_ms + 1
        last_ts_ms = ts_ms
        st.session_state["mp_last_ts_ms"] = last_ts_ms

        face_res = face_landmarker.detect_for_video(mp_image, ts_ms)
        hand_res = hand_landmarker.detect_for_video(mp_image, ts_ms)

        # Only predict when face landmarks are present (matches your training/inference logic).
        if not face_res.face_landmarks:
            stframe.image(frame, channels="BGR")
            continue

        face_features = _face_to_features(face_res.face_landmarks[0])

        left_features = [0.0] * (FEATURES_HAND)
        right_features = [0.0] * (FEATURES_HAND)

        if hand_res.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_res.hand_landmarks):
                handedness_name = None
                if hand_res.handedness and i < len(hand_res.handedness) and hand_res.handedness[i]:
                    handedness_name = hand_res.handedness[i][0].category_name
                if handedness_name == "Left":
                    left_features = _hand_to_features(hand_landmarks)
                elif handedness_name == "Right":
                    right_features = _hand_to_features(hand_landmarks)

        feats = face_features + left_features + right_features

        # Mediapipe landmark counts can differ slightly across models/versions.
        # The neural net expects a fixed input length, so we adapt the feature
        # vector to match the trained model shape.
        target_len = int(model.input_shape[-1])
        if len(feats) != target_len:
            if len(feats) > target_len:
                feats = feats[:target_len]
            else:
                feats = feats + [0.0] * (target_len - len(feats))

        x = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        pred_idx = int(np.argmax(model.predict(x, verbose=0)))
        predicted_labels.append(str(label[pred_idx]))

        cv2.putText(
            frame,
            f"Emotion: {predicted_labels[-1]}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        stframe.image(frame, channels="BGR")

    cap.release()

    if len(predicted_labels) < MIN_VALID_PREDICTIONS:
        st.warning(
            "Could not collect enough stable frames. Keep your face visible and try again."
        )
    else:
        counts = Counter(predicted_labels)
        top_emotion, top_count = counts.most_common(1)[0]
        majority_ratio = top_count / len(predicted_labels)
        if majority_ratio >= MIN_MAJORITY_RATIO:
            st.session_state["emotion"] = top_emotion
            st.success(f"Detected emotion: {top_emotion}")
        else:
            # Use the best guess so users can proceed even on noisy webcams.
            st.session_state["emotion"] = top_emotion
            st.warning(
                f"Detection was somewhat unstable; using best guess: {top_emotion}. "
                "For better accuracy, keep your face centered and well-lit."
            )

if st.session_state["emotion"] and st.button("🎶 Recommend Songs"):
    search_query = f"{lang} {st.session_state['emotion']} song {singer}"
    webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
    st.session_state["emotion"] = ""