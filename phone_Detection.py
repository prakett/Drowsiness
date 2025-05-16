from ultralytics import YOLO
import cv2

# Load YOLO model once
yolo_model = YOLO("yolo11s.pt")  # Replace with your custom model path

def detect_phone(frame):
    results = yolo_model(frame)[0]
    phone_alert = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]

        if label == "cell phone":  # Ensure this matches your model’s label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, "Phone Detected", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            phone_alert = True

    return phone_alert, frame
