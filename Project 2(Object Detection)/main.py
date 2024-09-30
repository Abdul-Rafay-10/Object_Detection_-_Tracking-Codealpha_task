
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Load the YOLOv8n model

# Load class names
with open('coco.names', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is typically the default camera

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference
    results = model(frame)

    # Draw results on the frame
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[int(cls)]}: {conf:.2f}"  # Use class name
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('YOLOv8 Live Object Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

