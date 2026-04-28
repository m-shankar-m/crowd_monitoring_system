import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Train YOLO for Crowd Counting\n",
            "This notebook uses the `ultralytics` package to train our crowd detection model on the prepared ShanghaiTech datasets.\n",
            "We initialize the model with `yolov8n.pt` and train on the `crowd_dataset.yaml` definition."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from ultralytics import YOLO\n",
            "import os\n",
            "\n",
            "os.makedirs('models/cv_weights', exist_ok=True)\n",
            "yaml_path = os.path.abspath('../data/yolo_dataset/crowd_dataset.yaml')\n",
            "\n",
            "print(f\"Using dataset config: {yaml_path}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize YOLOv8 Nano model (recommended for baseline training)\n",
            "model = YOLO('yolov8n.pt')\n",
            "\n",
            "# Train the model for 30 epochs (adjust based on hardware capabilities)\n",
            "results = model.train(data=yaml_path, epochs=30, imgsz=640, batch=16)\n",
            "\n",
            "print('Training complete!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import shutil\n",
            "\n",
            "# Copy best weights to models directory\n",
            "best_weights = 'runs/detect/train/weights/best.pt'\n",
            "dest_path = 'models/cv_weights/yolov8_crowd.pt'\n",
            "\n",
            "if os.path.exists(best_weights):\n",
            "    shutil.copy(best_weights, dest_path)\n",
            "    print(f'Model weights successfully saved to {dest_path}')\n",
            "else:\n",
            "    print('Could not find best.pt. Check ultralytics run directory.')\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "d:/VS Code/crowd prodiction/crowd_monitoring_system/notebooks/train_yolo.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {out_path}")
