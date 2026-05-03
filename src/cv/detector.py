import os
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='models/cv_weights/yolov8n.pt'):
        self.model = YOLO(model_path if os.path.exists(model_path) else 'yolov8n.pt')
            
    def detect(self, image):
        results = self.model(image, classes=[0], imgsz=640, conf=0.15, iou=0.45, verbose=False) 
        boxes = []
        if len(results) > 0:
            for box in results[0].boxes:
                # box: [x1, y1, x2, y2, conf, cls]
                b = box.xyxy[0].cpu().numpy()
                boxes.append(b)
        return boxes
