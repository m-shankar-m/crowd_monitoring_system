import os
from ultralytics import YOLO

class PersonDetector:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PersonDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path='models/cv_weights/yolov8n.pt'):
        if self._initialized:
            return
        self.model = YOLO(model_path if os.path.exists(model_path) else 'yolov8n.pt')
        self._initialized = True
            
    def detect(self, image):
        import torch
        with torch.no_grad():
            results = self.model(image, classes=[0], imgsz=640, conf=0.05, iou=0.45, verbose=False) 
            boxes = []
            if len(results) > 0:
                for box in results[0].boxes:
                    # box: [x1, y1, x2, y2, conf, cls]
                    b = box.xyxy[0].cpu().numpy()
                    boxes.append(b)
            return boxes
