import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv

GESTURE_NAME = "nod"

TARGET_SAMPLES = 700        
DATASET_DIR = "dataset"
CSV_PATH = os.path.join(DATASET_DIR, f"{GESTURE_NAME}.csv")
SAVE_KEY = ord('s')         
QUIT_KEY = ord('q')         

HAND_LANDMARKER_MODEL_PATH = r"D:\PIYUSH\dtu\aims\project_drone\gesture_using_mediapipe_and_ANN\hand_landmarker.task"

os.makedirs(DATASET_DIR, exist_ok=True)

current_samples = 0
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        current_samples = sum(1 for _ in reader)
    print(f"Found existing file {CSV_PATH} with {current_samples} samples.")
else:
    print(f"Creating new dataset file: {CSV_PATH}")

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

def extract_landmarks(landmarks):
    
    data = []
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z])
    return data

print("Instructions:")
print(f"- Currently recording gesture: '{GESTURE_NAME}'")
print(f"- Show this gesture to the camera, hold it steady.")
print(f"- Press 's' to save a sample when the hand is clearly visible.")
print(f"- Press 'q' to quit.")
print(f"Target samples for '{GESTURE_NAME}': {TARGET_SAMPLES}")

with open(CSV_PATH, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        hand_detected = result.hand_landmarks and len(result.hand_landmarks) > 0

        if hand_detected:
            h, w, _ = frame.shape
            for lm in result.hand_landmarks[0]:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)

        cv2.putText(frame, f"Gesture: {GESTURE_NAME}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {current_samples}/{TARGET_SAMPLES}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, "Press 's' to save, 'q' to quit", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if not hand_detected:
            cv2.putText(frame, "No hand detected", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == QUIT_KEY:
            print("Quitting...")
            break

        if key == SAVE_KEY and hand_detected:
            landmarks = result.hand_landmarks[0]
            row = extract_landmarks(landmarks)

            if len(row) == 63:  
                writer.writerow(row)
                current_samples += 1
                print(f"Saved sample {current_samples}/{TARGET_SAMPLES}")
            else:
                print("Unexpected number of landmarks, sample not saved.")

            if current_samples >= TARGET_SAMPLES:
                print(f"Reached target of {TARGET_SAMPLES} samples for '{GESTURE_NAME}'.")
                break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
