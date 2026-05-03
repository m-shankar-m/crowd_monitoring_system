import os
import shutil
import scipy.io
import cv2

def process_dataset(raw_dir, out_dir):
    parts = ['part_A_final', 'part_B_final']
    splits = [('train_data', 'train'), ('test_data', 'val')]
    
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
        
    for part in parts:
        for raw_split, out_split in splits:
            img_dir = os.path.join(raw_dir, part, raw_split, 'images')
            gt_dir = os.path.join(raw_dir, part, raw_split, 'ground_truth')
            
            if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
                print(f"Directory not found, skipping {img_dir} or {gt_dir}")
                continue
                
            for filename in os.listdir(img_dir):
                if not filename.endswith('.jpg'):
                    continue
                    
                img_path = os.path.join(img_dir, filename)
                base_name = os.path.splitext(filename)[0]
                gt_path = os.path.join(gt_dir, f'GT_{base_name}.mat')
                
                # Check if GT exists
                if not os.path.exists(gt_path):
                    continue
                    
                # Load image to get dimensions
                img = cv2.imread(img_path)
                if img is None:
                    continue
                height, width, _ = img.shape
                
                # Copy image
                unique_name = f"{part}_{base_name}.jpg"
                out_img_path = os.path.join(out_dir, 'images', out_split, unique_name)
                shutil.copy(img_path, out_img_path)
                
                # Process GT
                mat = scipy.io.loadmat(gt_path)
                points = mat['image_info'][0, 0][0, 0][0]
                
                # Generate YOLO boxes
                # Using a fixed 30x30 bounding box (or clipped to image bounds)
                yolo_annotations = []
                box_w = 30 / width
                box_h = 30 / height
                
                for pt in points:
                    x, y = pt[0], pt[1]
                    # Normalize coordinates
                    x_center = min(max(x / width, 0.0), 1.0)
                    y_center = min(max(y / height, 0.0), 1.0)
                    
                    yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
                    
                # Save label
                label_name = f"{part}_{base_name}.txt"
                out_label_path = os.path.join(out_dir, 'labels', out_split, label_name)
                with open(out_label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))

    print(f"Dataset preparation complete. Saved to {out_dir}")
    
    # Generate yaml
    yaml_content = f"""path: {os.path.abspath(out_dir).replace(os.sep, '/')}
train: images/train
val: images/val

names:
  0: person
"""
    yaml_path = os.path.join(out_dir, "crowd_dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created yaml config at {yaml_path}")

if __name__ == "__main__":
    raw_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/ShanghaiTech_Crowd_Counting_Dataset'))
    yolo_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/yolo_dataset'))
    process_dataset(raw_dataset_dir, yolo_dataset_dir)
