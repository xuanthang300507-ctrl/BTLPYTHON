import cv2
from ultralytics import YOLO

model = YOLO("m1detect/anpr_v1/weights/best.pt")

def detect_and_crop_plate(image_path):
    results = model.predict(image_path, conf=0.4, verbose=False)[0]

    if len(results.boxes) == 0:
        return None

    img = cv2.imread(image_path)

    best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop