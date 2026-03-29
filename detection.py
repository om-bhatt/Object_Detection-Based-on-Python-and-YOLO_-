from ultralytics import YOLO
import cv2
import time

# Load model (better accuracy)
model = YOLO("yolov8s.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# FPS
prev_time = 0

# Object counter
object_count = {}

# Target objects (customize this 🔥)
target_objects = ["person", "cell phone"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    object_count.clear()

    # Detection
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Filter only selected objects
            if label in target_objects:

                # Count objects
                if label not in object_count:
                    object_count[label] = 0
                object_count[label] += 1

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # 🚨 Alert (example)
                if label == "person":
                    cv2.putText(frame, "ALERT: PERSON DETECTED!",
                                (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

    # 🔢 Show counts
    y_offset = 120
    for obj, count in object_count.items():
        cv2.putText(frame, f"{obj}: {count}",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        y_offset += 30

    # ⚡ FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Window title
    cv2.imshow("🔥 Advanced YOLO Detection System", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()