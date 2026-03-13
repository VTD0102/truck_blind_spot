from collections import Counter
import os

counter = Counter()

label_dir = "yolov9/data/datasets/train/labels"

for file in os.listdir(label_dir):

    with open(os.path.join(label_dir,file)) as f:
        for line in f:
            cls = int(line.split()[0])
            counter[cls] += 1

print(counter)