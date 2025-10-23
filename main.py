"""
Multi-Object Detection and Tracking using YOLOv11 TensorRT model and DeepSORT with Gradio UI.
"""
import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import gradio as gr
import time
import tempfile


class ObjectDetectionAndTracking:
    def __init__(self):
        self.model = YOLO('/home/agent/Downloads/akilan/runs/detect/train/weights/best.engine')         # Load YOLOv11 TensorRT model
        self.tracker = DeepSort(max_age=30)                                                             # Initialize DeepSORT tracker
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']

    def process_image(self, file):
        """ 
        Process a single image for object detection and return annotated image and stats.
        """
        image = cv2.imread(file)
        start = time.time()
        results = self.model.predict(source=image, conf=0.5)                                            # Detect objects  
        end = time.time()

        detected_counts = {name: 0 for name in self.class_names}
        for box in results[0].boxes:                                                                    # Draw bounding boxes
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{self.class_names[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if cls < len(self.class_names):
                detected_counts[self.class_names[cls]] += 1

        stats_text = '\n'.join([f"{k}: {v}" for k, v in detected_counts.items()])
        stats_text += f"\nDetection Time: {(end - start) * 1000:.1f} ms"
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), stats_text

    def process_video(self, video_file):
        """
        Process a video file for object detection and tracking, return annotated video.
        """
        cap = cv2.VideoCapture(video_file.name if hasattr(video_file, "name") else video_file)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = orig_fps if orig_fps and orig_fps > 1 else 20
        frames = []
        chunk_idx = 0
        chunk_size = int(fps) # ~1 second per chunk

        prev_time = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            height, width, _ = frame.shape

            # Yolo detection
            results = self.model.predict(source=frame, conf=0.5)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(([x1, y1, w, h], conf, cls))

            # DeepSORT tracking
            tracks = self.tracker.update_tracks(detections, frame=frame)
            active_ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                active_ids.append(tid)

            # FPS calculation and stats overlay
            curr_time = cv2.getTickCount()
            if prev_time is None:
                inst_fps = 0.0
            else:
                inst_fps = cv2.getTickFrequency() / (curr_time - prev_time)
            prev_time = curr_time

            stats_text = f"FPS: {inst_fps:.2f} | Active Tracks: {len(active_ids)}"
            text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x = width - text_size[0] - 20
            y = height - 20
            cv2.putText(frame, stats_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            frames.append(frame)

            # For streaming, yield video chunks
            if len(frames) == chunk_size or not ret:
                tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out = cv2.VideoWriter(tmp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                for f in frames:
                    out.write(f)
                out.release()
                yield tmp_video.name
                os.unlink(tmp_video.name)
                frames = []
                chunk_idx += 1

        cap.release()

    def launch_demo(self):
        """
        Launch Gradio Interface.
        """
        iface_image = gr.Interface(
                                fn = self.process_image,
                                inputs = gr.File(label="Upload Image"),
                                outputs = [gr.Image(label="Detection Output"), gr.Textbox(label="Detection Stats:", lines=4)],
                                title = "Real-time Object Detection and Tracking with YOLOv11 + DeepSORT"
                                )

        iface_video = gr.Interface(
                                fn = self.process_video,
                                inputs = gr.File(label="Upload Video"),
                                outputs = gr.Video(label="Detection + Tracking", streaming=True, autoplay=True),
                                title = "Real-time Object Detection and Tracking with YOLOv11 + DeepSORT"
                                )

        demo = gr.TabbedInterface([iface_image, iface_video], ["Image Input", "Video Input"])
        demo.launch()


if __name__ == "__main__":
    app = ObjectDetectionAndTracking()
    app.launch_demo()
