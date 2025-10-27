"""
Evaluate Multi-Object Tracking on KITTI dataset.

Steps:
1. Get ground truth from KITTI labels.
2. Get detections from YOLOv11 model.
3. Get tracks from DeepSort.
4. Compute IoU motmetrics - MOTA, MOTP, IDF1.
"""

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import motmetrics as mm
from dotenv import load_dotenv
from helper import load_kitti_gt_label, iou_distance

load_dotenv()
user_dir = os.getenv("USER_DIR")

# --- CONFIGURATION ---
kitti_seq = '0016'  # Sequence to test
kitti_image_dir = os.path.join(user_dir, f"data_tracking_image_2/training/image_02/{kitti_seq}")
kitti_label_path = os.path.join(user_dir, "data_tracking_label_2/training/label_02/{kitti_seq}.txt")
yolo_weights = os.path.join(user_dir, "runs/detect/train/weights/best.engine")

model = YOLO(yolo_weights)              # Load YOLOv11 model
tracker = DeepSort(max_age=30)          # Initialize DeepSort tracker
acc = mm.MOTAccumulator(auto_id=True)   # Initialize MOT accumulator

# --- LOAD GROUND TRUTH ---
gt_dict = load_kitti_gt_label(kitti_label_path)

# --- RUN OVER IMAGE SEQUENCE ---
img_files = sorted(glob.glob(os.path.join(kitti_image_dir, '*.png')))
for frame_id, img_path in enumerate(img_files):
    frame = cv2.imread(img_path)
    results = model.predict(source=frame, conf=0.5)
    detections_for_tracker = []

    # Prepare detections
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections_for_tracker.append(([x1, y1, w, h], conf, cls))

    # Tracker update
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # Gather tracker boxes and IDs
    track_ids, track_boxes = [], []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_ids.append(track.track_id)
        track_boxes.append(list(track.to_ltrb()))  # x1, y1, x2, y2
    track_boxes = np.array(track_boxes) if track_boxes else np.empty((0, 4))

    # Gather GT
    gt_annos = gt_dict.get(frame_id, [])
    gt_ids = [tid for tid, box in gt_annos]
    gt_boxes = [box for tid, box in gt_annos]
    gt_boxes = np.array(gt_boxes) if gt_boxes else np.empty((0, 4))

    # Update MOT accumulator
    cost_matrix = iou_distance(gt_boxes, track_boxes)
    acc.update(gt_ids, track_ids, cost_matrix)

    # visualization
    for i, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID {track.track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# --- COMPUTE AND REPORT METRICS ---
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1'], name='KITTI_seq')
print('\nMulti-Object Tracking Evaluation Metrics on KITTI:')
print(summary)
