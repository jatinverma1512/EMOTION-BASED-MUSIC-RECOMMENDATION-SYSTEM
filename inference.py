import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
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
FEATURES_FACE = FACE_LANDMARK_COUNT * 2
FEATURES_HAND = HAND_LANDMARK_COUNT * 2
FEATURES_TOTAL = FEATURES_FACE + (FEATURES_HAND * 2)


def _ensure_mediapipe_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_MODEL_PATH.exists():
        urlretrieve(MODEL_FACE_URL, FACE_MODEL_PATH.as_posix())
    if not HAND_MODEL_PATH.exists():
        urlretrieve(MODEL_HAND_URL, HAND_MODEL_PATH.as_posix())


def _face_to_features(face_landmarks) -> list[float]:
    # Offset all face landmarks relative to landmark index 1.
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
    # Offset all hand landmarks relative to landmark index 8.
    origin = hand_landmarks[8]
    feats: list[float] = []
    for lm in hand_landmarks:
        feats.append(lm.x - origin.x)
        feats.append(lm.y - origin.y)
    return feats


model = load_model("model.h5")
label = np.load("labels.npy")

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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not detected. Please check your webcam.")

t0_ms = int(time.time() * 1000)

while True:
    ret, frm = cap.read()
    if not ret:
        print("Unable to read from camera")
        break

    frm = cv2.flip(frm, 1)
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms = int(time.time() * 1000) - t0_ms

    face_res = face_landmarker.detect_for_video(mp_image, ts_ms)
    hand_res = hand_landmarker.detect_for_video(mp_image, ts_ms)

    if not face_res.face_landmarks:
        cv2.imshow("window", frm)
        if cv2.waitKey(1) == 27:
            break
        continue

    face_features = _face_to_features(face_res.face_landmarks[0])
    left_features = [0.0] * FEATURES_HAND
    right_features = [0.0] * FEATURES_HAND

    if hand_res.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_res.hand_landmarks):
            handedness_name = None
            if (
                hand_res.handedness
                and i < len(hand_res.handedness)
                and hand_res.handedness[i]
            ):
                handedness_name = hand_res.handedness[i][0].category_name
            if handedness_name == "Left":
                left_features = _hand_to_features(hand_landmarks)
            elif handedness_name == "Right":
                right_features = _hand_to_features(hand_landmarks)

    feats = face_features + left_features + right_features

    target_len = int(model.input_shape[-1])
    if len(feats) != target_len:
        # Mediapipe landmark counts can vary slightly; the neural net expects
        # a fixed input size, so we truncate/pad deterministically.
        if len(feats) > target_len:
            feats = feats[:target_len]
        else:
            feats = feats + [0.0] * (target_len - len(feats))

    x = np.asarray(feats, dtype=np.float32).reshape(1, -1)
    pred_idx = int(np.argmax(model.predict(x, verbose=0)))
    pred = label[pred_idx]
    cv2.putText(frm, str(pred), (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    print(pred)

    cv2.imshow("window", frm)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

