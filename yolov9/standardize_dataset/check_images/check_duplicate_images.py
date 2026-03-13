import os
import hashlib

def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

hashes = set()

for img in os.listdir("yolov9/data/datasets/train/images"):
    h = file_hash(f"yolov9/data/datasets/train/images/{img}")
    if h in hashes:
        print("Duplicate:", img)
    hashes.add(h)