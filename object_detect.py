# for detecting in videos:
import cv2
from ultralytics import YOLO

# model load:
y_model = YOLO('./model/yolov8n.pt')

# video path:
#video_path = './input/input4.mp4'
# load video
cap = cv2.VideoCapture(1)

# main program loop:
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame resizing:
    frame = cv2.resize(frame, (640, 480))
    # initialize yolo algorithm:
    # confidence/conf is high confidence threshold
    # iou is only detect more accurate results
    results = y_model(frame, conf=0.5, iou=0.4)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            label = f"{y_model.names[class_id]} {confidence:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # show the result frame:
    cv2.imshow("Drone footage001", frame)

    # exit on 'q' key:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release:
cap.release()
cv2.destroyAllWindows()
