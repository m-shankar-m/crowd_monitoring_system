import math

class Tracker:
    def __init__(self, dist_threshold=120, max_disappeared=30):
        self.objects = {} 
        self.disappeared = {}
        self.next_id = 0
        self.dist_threshold = dist_threshold
        self.max_disappeared = max_disappeared

    def update(self, boxes):
        # Current centroids
        input_centroids = []
        for box in boxes:
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            input_centroids.append((cx, cy, box))

        if not self.objects:
            for cx, cy, box in input_centroids:
                self.register(cx, cy)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            used_inputs = set()
            used_objects = set()
            
            # Simple matching logic
            for i, (ocx, ocy) in enumerate(object_centroids):
                best_dist = self.dist_threshold
                best_idx = -1
                for j, (icx, icy, _) in enumerate(input_centroids):
                    if j in used_inputs: continue
                    dist = math.hypot(icx - ocx, icy - ocy)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = j
                
                if best_idx != -1:
                    obj_id = object_ids[i]
                    self.objects[obj_id] = (input_centroids[best_idx][0], input_centroids[best_idx][1])
                    self.disappeared[obj_id] = 0
                    used_inputs.add(best_idx)
                    used_objects.add(i)

            # Mark missing objects
            for i in range(len(object_ids)):
                if i not in used_objects:
                    obj_id = object_ids[i]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)

            # Register new objects
            for j, (icx, icy, _) in enumerate(input_centroids):
                if j not in used_inputs:
                    self.register(icx, icy)

        # Build tracks for current frame display
        tracks = []
        # Match current boxes to current object IDs for drawing
        for obj_id, (ocx, ocy) in self.objects.items():
            # Find closest input box to this object
            best_dist = 50 # tight match for drawing
            best_box = None
            for cx, cy, box in input_centroids:
                dist = math.hypot(cx - ocx, cy - ocy)
                if dist < best_dist:
                    best_dist = dist
                    best_box = box
            
            if best_box is not None:
                tracks.append({"id": obj_id, "bbox": best_box[:4]})
            
        return tracks

    def register(self, cx, cy):
        self.objects[self.next_id] = (cx, cy)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
