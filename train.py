from ultralytics import YOLO
import torch

# Kiểm tra GPU
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# -------------------------------------------------------------
# CẤU HÌNH RESUME: Chỉ định file checkpoint gần nhất trước khi sập
# Thường nằm ở: project/name/weights/last.pt
# -------------------------------------------------------------
checkpoint_last = "LAPTRINH PYTHON BTL /Dữ liệu train sau khi scale khuônn ảnh/weights/last.pt" 

try:
    # Thử load lại model từ checkpoint cũ để chạy tiếp
    model = YOLO(checkpoint_last)
    print("Tìm thấy checkpoint cũ. Đang chuẩn bị resume...")
    
    results = model.train(
        resume=True,  # BẮT BUỘC: Bật True để YOLO hiểu là train tiếp
        device="cpu"  # Đảm bảo giữ nguyên thiết bị train
    )
except Exception as e:
    # Nếu không tìm thấy file last.pt (hoặc file bị lỗi do sập nguồn đột ngột), train lại từ đầu
    print(f"Không thể resume ({e}). Tiến hành train lại từ đầu...")
    
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="vietnamese-license-plate-1/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,              
        patience=20,           
        device="cpu",          
        optimizer="AdamW",
        lr0=0.001,
        augment=True,         
        project="runs",
        name="anpr_v1",
        save=True,
        plots=True             
    )

print(f"\n Train hoàn thiện!")
print(f"Best model: LAPTRINH PYTHON BTL /Dữ liệu train sau khi scale khuônn ảnh/weights/best.pt")