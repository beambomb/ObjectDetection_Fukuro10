from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")  # Pastikan best.pt ada di folder yang sama

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    for result in results:
        for box in result.boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Class ID -> label
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            # Centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Draw centroid
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Text label + centroid coords
            cv2.putText(frame, f"{label} ({cx}, {cy})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("YOLOv8 Webcam + Centroid", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
