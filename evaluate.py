# evaluate.py
from ultralytics import YOLO

model = YOLO("runs/anpr_v1/weights/best.pt")

# Đánh giá trên tập val
metrics = model.val(data="data.yaml")
print(f"mAP@0.5:     {metrics.box.map50:.4f}")   # mục tiêu > 0.85
print(f"mAP@0.5:0.95:{metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")