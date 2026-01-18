# Real-time Object Detection and Tracking: YOLOv11 TensorRT model and DeepSORT
The objective of this project is to develop a real-time object detection and tracking model for self-driving cars and autonomous robots.

### KITTI Dataset:
[KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) is a popular benchmark dataset for Computer Vision and Machine Learning applications. In total the dataset contain 14,999 images with 7,481 train and 7,518 test images. The training set is annotated with ground-truth labels.<br/>
The annotated labels have a specific format: `<type> <truncated> <occluded> <alpha> <bbox_left> <bbox_top> <bbox_right> <bbox_bottom> <height> <width> <length> <x> <y> <z> <rotation_y> <score>`.<br/>
There are 9 label categories: `Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc, DontCare`. But I focus on the 3 main classes: `Car, Pedestrian, Cyclist`.

### Solution:
I use [YOLOv11 model](https://docs.ultralytics.com/models/yolo11/) for object detection and DeepSORT for object tracking.
1. <ins>Dataset Preparation</ins>: Convert KITTI labels to YOLO label format. YOLO label format: `<class_id> <x_center> <y_center> <width> <height>`. KITTI training images are then split into training and validation sets.
2. <ins>Model Training</ins>: 

