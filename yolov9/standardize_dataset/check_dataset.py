import cv2
import os

img_path = "yolov9/data/datasets/train/images"
label_path = "yolov9/data/datasets/train/labels"

for file in os.listdir(img_path):

    img = cv2.imread(os.path.join(img_path,file))

    h, w, _ = img.shape

    label_file = file.replace(".jpg",".txt")

    with open(os.path.join(label_path,label_file)) as f:

        for line in f:
            cls, x, y, bw, bh = map(float,line.split())

            x1 = int((x-bw/2)*w)
            y1 = int((y-bh/2)*h)
            x2 = int((x+bw/2)*w)
            y2 = int((y+bh/2)*h)

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("bbox", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()