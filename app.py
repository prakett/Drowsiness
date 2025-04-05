import torch
import cv2
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import mediapipe as mp
import time

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load("drowsiness_detector_model.pth", map_location=device))
model.eval().to(device)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Drowsiness detection logic
drowsy_frame_count = 0
awake_frame_count = 0
DROWSY_THRESHOLD = 30              # Drowsy if eyes closed for 30 frames
AWAKE_RESET_THRESHOLD = 5          # Only reset drowsy counter after 5 awake frames

# === IP CAM SETUP ===
ip_cam_url = "http://192.168.0.108:4747/video"  # Replace with your phone's IP camera stream URL
cap = cv2.VideoCapture(ip_cam_url)

# FPS tracking
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    label = "AWAKE"  # Default label
    color = (0, 255, 0)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            w_box = int(bboxC.width * w)
            h_box = int(bboxC.height * h)

            # Crop face
            face_crop = frame[y:y + h_box, x:x + w_box]
            if face_crop.size == 0:
                continue

            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor).squeeze()
                prob = torch.sigmoid(output).item()
                is_drowsy = prob < 0.5
                print(f"Prediction: {prob:.3f}")

            # Frame counter logic
            if is_drowsy:
                drowsy_frame_count += 1
                awake_frame_count = 0
            else:
                awake_frame_count += 1
                if awake_frame_count >= AWAKE_RESET_THRESHOLD:
                    drowsy_frame_count = 0

            # Decide final label
            if drowsy_frame_count >= DROWSY_THRESHOLD:
                label = "DROWSY"
                color = (0, 0, 255)
            else:
                label = "AWAKE"
                color = (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
