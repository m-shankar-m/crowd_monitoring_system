import math

class Tracker:
    def __init__(self, dist_threshold=100):
        self.objects = {} 
        self.next_id = 0
        self.dist_threshold = dist_threshold

    def update(self, boxes):
        tracks = []
        new_objects = {}
        for box in boxes:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            best_id = -1
            best_dist = float('inf')
            
            for obj_id, old_centroid in self.objects.items():
                old_cx, old_cy = old_centroid
                dist = math.hypot(cx - old_cx, cy - old_cy)
                if dist < self.dist_threshold and dist < best_dist:
                    best_dist = dist
                    best_id = obj_id
                    
            if best_id != -1:
                new_objects[best_id] = (cx, cy)
                tracks.append({"id": best_id, "bbox": box[:4]})
                del self.objects[best_id]
            else:
                new_objects[self.next_id] = (cx, cy)
                tracks.append({"id": self.next_id, "bbox": box[:4]})
                self.next_id += 1
                
        self.objects = new_objects
        return tracks
