from src.cv.pipeline import CVPipeline
import cv2
import numpy as np

cv_pipeline = CVPipeline()

def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image or empty payload")
        
    return cv_pipeline.process_frame(frame)
