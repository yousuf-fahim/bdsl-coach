"""Phase 0: record the software/hardware environment to results/environment.json.

The report cites this file instead of hand-typing versions or hardware.
Run: .venv\\Scripts\\python src\\check_env.py
"""
import json
import os
import platform
import sys
from pathlib import Path

import cv2
import imagehash
import matplotlib
import mediapipe
import numpy
import onnx
import onnxruntime
import pandas
import PIL
import seaborn
import sklearn
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]

info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "cpu": platform.processor(),
    "logical_cpus": os.cpu_count(),
    "torch_threads": torch.get_num_threads(),
    "cuda_available": torch.cuda.is_available(),
    "versions": {m.__name__: m.__version__ for m in (
        numpy, pandas, sklearn, matplotlib, seaborn, cv2, mediapipe, torch,
        torchvision, onnx, onnxruntime, imagehash, PIL)},
}

out = ROOT / "results" / "environment.json"
out.write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2))
