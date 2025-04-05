import torch
import cv2
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import mediapipe as mp
import time
from collections import deque

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load("drowsiness_detector_model.pth", map_location=device))
model.eval().to(device)

# MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Camera
cap = cv2.VideoCapture(1)

# FPS
prev_time = time.time()

# Prediction smoothing
prediction_history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    label = "AWAKE"
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
                prediction_history.append(prob)

                avg_prob = sum(prediction_history) / len(prediction_history)
                is_drowsy = avg_prob < 0.6

                print(f"Raw: {prob:.3f}, Avg: {avg_prob:.3f}, Drowsy: {is_drowsy}")

            if is_drowsy:
                label = "DROWSY"
                color = (0, 0, 255)
            else:
                label = "AWAKE"
                color = (0, 255, 0)

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
