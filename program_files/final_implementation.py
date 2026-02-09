import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import tensorflow as tf
from collections import deque

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_ann.h5")
NORM_PATH = os.path.join(MODELS_DIR, "normalization.npz")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.txt")

CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5  

HAND_LANDMARKER_MODEL_PATH = r"D:\PIYUSH\dtu\aims\project_drone\gesture_using_mediapipe_and_ANN\hand_landmarker.task"

model = tf.keras.models.load_model(MODEL_PATH)


norm_data = np.load(NORM_PATH)
MEAN = norm_data["mean"]
STD = norm_data["std"]
FEATURE_DIM = MEAN.shape[0]

# Load labels
with open(LABELS_PATH, "r") as f:
    GESTURES = [line.strip() for line in f if line.strip()]

# Map gesture label -> "drone command" string
GESTURE_TO_COMMAND = {
    "up":           "MOVE UP",
    "down":         "MOVE DOWN",
    "left":         "MOVE LEFT",
    "right":        "MOVE RIGHT",
    "land":         "LAND",
    "backflip":     "BACKFLIP",
    "return_home":  "RETURN HOME",
    "spin_180":     "180 SPIN",
    "spin_360":     "360 SPIN",
    "takeoff":      "TAKEOFF",
    "forward_up":   "MOVE FORWARD AND UP",
    "forward_down": "MOVE FORWARD AND DOWN",
    "nod":          "NOD",
}

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

base_options = BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=VisionRunningMode.IMAGE 
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)


pred_history = deque(maxlen=SMOOTHING_WINDOW)

def extract_landmarks(landmarks):
    
    data = []
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z])
    return data

def compute_extra_features(coords):

    fingertips_idx = [4, 8, 12, 16, 20]
    fingertip_dists = [np.linalg.norm(coords[i]) for i in fingertips_idx]

    thumb_tip = coords[4]
    other_tips = [coords[i] for i in [8, 12, 16, 20]]
    thumb_pair_dists = [np.linalg.norm(thumb_tip - tip) for tip in other_tips]

    return np.array(fingertip_dists + thumb_pair_dists, dtype=np.float32)

def preprocess_sample(sample_vec):
    
    sample_vec = np.asarray(sample_vec, dtype=np.float32)
    if sample_vec.shape[0] != 63:
        raise ValueError(f"Expected 63 raw features, got {sample_vec.shape[0]}")

    coords = sample_vec.reshape(21, 3)
    wrist = coords[0].copy()
    coords -= wrist

    dists = np.linalg.norm(coords, axis=1)
    max_dist = np.max(dists)
    if max_dist > 0:
        coords /= max_dist

    extra = compute_extra_features(coords)
    coords_flat = coords.flatten()
    full_features = np.concatenate([coords_flat, extra], axis=0)

    if full_features.shape[0] != FEATURE_DIM:
        raise ValueError(
            f"Processed feature dim {full_features.shape[0]} "
            f"does not match trained dim {FEATURE_DIM}"
        )
    return full_features

print("Real-time gesture control started.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    command_text = "No command detected / Hover"
    display_text = "No hand detected"

    if result.hand_landmarks and len(result.hand_landmarks) > 0:
        landmarks = result.hand_landmarks[0]

        h, w, _ = frame.shape
        for lm in landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)

        features = extract_landmarks(landmarks)

        try:
            x = preprocess_sample(np.array(features, dtype=np.float32))
            x_norm = (x - MEAN) / (STD + 1e-6)
            x_norm = np.expand_dims(x_norm, axis=0) 

            probs = model.predict(x_norm, verbose=0)[0] 

            pred_history.append(probs)

            avg_probs = np.mean(pred_history, axis=0)
            pred_idx = int(np.argmax(avg_probs))
            pred_conf = float(avg_probs[pred_idx])
            pred_label = GESTURES[pred_idx] if pred_idx < len(GESTURES) else "unknown"

            if pred_conf >= CONFIDENCE_THRESHOLD and pred_label in GESTURE_TO_COMMAND:
                cmd = GESTURE_TO_COMMAND[pred_label]
                command_text = f"DRONE: {cmd}"
                display_text = f"{pred_label} ({pred_conf:.2f})"
                print(f"Gesture: {pred_label}, Confidence: {pred_conf:.2f}, Command: {command_text}")
            else:
                command_text = "No command detected / Hover"
                display_text = "Low confidence"
        except ValueError as e:
            display_text = str(e)
    else:
        pred_history.clear()

    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, command_text, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Threshold: {CONFIDENCE_THRESHOLD:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Drone Gesture Control (Simulation)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

landmarker.close()
