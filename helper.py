"""
Helper functions.
"""

import os
import numpy as np

# Helper to convert KITTI labels to YOLO format
def convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir, class_map,image_size):
    """
    Convert KITTI labels to YOLO format.
    KITTI format: <class> <truncated> <occluded> <alpha> <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> ...
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    """
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)

    for label_file in os.listdir(kitti_label_dir):
        label_path = os.path.join(kitti_label_dir, label_file)
        yolo_file_path = os.path.join(yolo_label_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        with open(yolo_file_path, 'w') as yolo_file:
            for line in lines:
                line_parts = line.split()
                class_name = line_parts[0]

                if class_name not in class_map or class_map[class_name] == -1:
                    continue

                class_id = class_map[class_name]
                # Extract bounding box coordinates (KITTI format)
                xmin = float(line_parts[4])
                ymin = float(line_parts[5])
                xmax = float(line_parts[6])
                ymax = float(line_parts[7])

                # Convert to YOLO format
                image_width, image_height = image_size
                x_center = (xmin + xmax) / 2 / image_width
                y_center = (ymin + ymax) / 2 / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height

                # Write the annotation in YOLO format
                yolo_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Converted {label_file} to YOLO format.")


# Helper to load KITTI GT labels for a sequence
def load_kitti_gt_label(label_path, valid_classes=['Car', 'Pedestrian', 'Cyclist']):
    frame_data = {}
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        frame_id = int(parts[0])
        track_id = int(parts[1])
        obj_class = parts[2]
    
        if obj_class not in valid_classes:
            continue
    
        bbox = [float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])]
        frame_data.setdefault(frame_id, []).append((track_id, bbox))
    
    return frame_data


# IoU distance function for motmetrics
def iou_distance(gt, hyp):
    def iou(a, b):
        xi1, yi1, xi2, yi2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        union_area = a_area + b_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    if len(gt) == 0 or len(hyp) == 0:
        return np.empty((len(gt), len(hyp)))
    
    cost_matrix = np.zeros((len(gt), len(hyp)))
    
    for i in range(len(gt)):
        for j in range(len(hyp)):
            cost_matrix[i, j] = 1 - iou(gt[i], hyp[j])
    
    return cost_matrix