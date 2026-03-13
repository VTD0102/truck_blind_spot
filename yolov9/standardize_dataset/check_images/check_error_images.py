import cv2
import os

for img in os.listdir("yolov9/data/datasets/train/images"):
    path = f"yolov9/data/datasets/train/images/{img}"
    if cv2.imread(path) is None:
        print("Corrupt image:", img)