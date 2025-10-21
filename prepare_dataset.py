"""
Prepare Dataset.

Steps:
1. Convert KITTI labels to YOLO format.
2. Split into training/validation sets.
"""

import os
import shutil
import random
from dotenv import load_dotenv
from helper import convert_kitti_to_yolo

load_dotenv()
user_dir = os.getenv("USER_DIR")

# --- Convert KITTI labels to YOLO format ---
class_map = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'DontCare': -1
}

kitti_label_dir = os.path.join(user_dir, "data_object_label_2/training/label_2")
yolo_label_dir = os.path.join(user_dir, "yolo/labels")
image_size = (1242, 375)

convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir, class_map, image_size)

# --- Split dataset into training and validation sets ---
# Directories
image_dir = os.path.join(user_dir, "data_object_image_2/training/image_2")
label_dir = os.path.join(user_dir, "yolo/labels")
train_image_dir = os.path.join(user_dir, "dataset/images/train")
val_image_dir = os.path.join(user_dir, "dataset/images/val")
train_label_dir = os.path.join(user_dir, "dataset/labels/train")
val_label_dir = os.path.join(user_dir, "dataset/labels/val")

# Get all training image files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Shuffle and split data into train and validation sets (80-20 split)
random.shuffle(image_files)
split_idx = int(0.8 * len(image_files))

train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# Move training images and labels
for file in train_files:
    shutil.copy(os.path.join(image_dir, file), os.path.join(train_image_dir, file))
    label_file = file.replace('.png', '.txt').replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

# Move validation images and labels
for file in val_files:
    shutil.copy(os.path.join(image_dir, file), os.path.join(val_image_dir, file))
    label_file = file.replace('.png', '.txt').replace('.jpg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))

print("Data Split Completed")