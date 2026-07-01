from ultralytics import YOLO
model1 = YOLO("m1detect/anpr_v1/weights/best.pt")
metrics1 = model1.val(data="vietnamese-license-plate-1/data.yaml")

print(f"mAP@0.5:      {metrics1.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics1.box.map:.4f}")
print(f"Precision:    {metrics1.box.mp:.4f}")
print(f"Recall:       {metrics1.box.mr:.4f}")