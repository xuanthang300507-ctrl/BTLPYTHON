from ultralytics import YOLO
model = YOLO("LAPTRINH PYTHON BTL/Dữ liệu train sau khi scale khuônn ảnh/weights/best.pt")

# Đánh giá trên tập dữ liệu validation (data.yaml)
metrics = model.val(data="LAPTRINH PYTHON BTL/vietnamese-license-plate-1/data.yaml")
print(f"mAP@0.5:     {metrics.box.map50:.4f}")   # mục tiêu > 0.85
print(f"mAP@0.5:0.95:{metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")


