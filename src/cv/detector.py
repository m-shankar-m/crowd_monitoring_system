import os
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="models/cv_weights/yolov8n.pt"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model = YOLO(model_path if os.path.exists(model_path) else 'yolov8n.pt')
        if not os.path.exists(model_path) and os.path.exists('yolov8n.pt'):
            import shutil
            shutil.move('yolov8n.pt', model_path)
            self.model = YOLO(model_path)
            
    def detect(self, image):
        # Reduced image size and switched to nano model to save RAM
        results = self.model(image, classes=[0], imgsz=640, conf=0.15, iou=0.45, verbose=False) 
        boxes = []
        if len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
        return boxes
