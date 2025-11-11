from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Confidence
            conf = float(box.conf[0])
            conf = round(conf, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            cv2.putText(frame, f"{label} {conf} ({cx}, {cy})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("YOLOv8 Webcam + Centroid", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

