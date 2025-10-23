"""
Train YOLOv11 Model
"""

from ultralytics import YOLO

model = YOLO('yolo11s.pt')                                                  # Load a pre-trained YOLOv11 model
model.train(data='dataset/dataset.yaml', epochs=50, imgsz=640, batch=16)    # Train the model with your dataset
