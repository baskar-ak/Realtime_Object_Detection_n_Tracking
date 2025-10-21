import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv11 model
model = YOLO('runs/detect/train/weights/best.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open video file
cap = cv2.VideoCapture('traffic.mp4')

cv2.namedWindow("Object Detection & Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection & Tracking", 1200, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv11 detections
    results = model.predict(source=frame, conf=0.25)
    detections_for_tracker = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections_for_tracker.append(([x1, y1, w, h], conf, cls))

    # Update DeepSORT tracker with detections and get tracks
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # Visualize tracked objects with bounding boxes and IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('YOLOv11 + DeepSORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
