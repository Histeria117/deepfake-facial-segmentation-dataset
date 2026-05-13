import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

print("Python:", sys.version)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("MediaPipe:", mp.__version__)

try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch no disponible o con error:", e)