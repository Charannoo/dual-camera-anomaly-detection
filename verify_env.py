import torch
import torchvision
import timm
import numpy as np
import sklearn
import cv2
import matplotlib
import PIL
import yaml
import tifffile

print("--- Environment Verification ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"timm Version: {timm.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"scikit-learn Version: {sklearn.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"Matplotlib Version: {matplotlib.__version__}")
print(f"Pillow Version: {PIL.__version__}")
print(f"PyYAML Version: {yaml.__version__}")
print(f"Tifffile Version: {tifffile.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    print("Running in CPU-only mode (as expected).")
print("Environment check PASSED.")
