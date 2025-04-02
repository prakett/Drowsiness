import torch
import cv2
import numpy as np
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
import mediapipe as mp

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 1)
model.load_state_dict(torch.load("drowsiness_detector_model.pth", map_location=device))
model.eval().to(device)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Define preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop face
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            # Convert face crop to PIL image
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

            # Preprocess image
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                output = model(face_tensor).squeeze()
                prob = torch.sigmoid(output).item()
                label = "AWAKE" if prob > 0.5 else "DROWSY"  # Fixed label logic

            # Display label
            color = (0, 0, 255) if label == "DROWSY" else (0, 255, 0)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the output frame
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()