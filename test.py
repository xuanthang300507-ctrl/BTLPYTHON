import cv2
from ultralytics import YOLO
import easyocr

# ==========================================
# KHỞI TẠO BƯỚC 1 & BƯỚC 2
# ==========================================
# 1. Tải Model 1 (YOLO11n của bạn)
model_1 = YOLO("LAPTRINH PYTHON BTL /Dữ liệu train sau khi scale khuônn ảnh/weights/best.pt") 

# 2. Khởi tạo EasyOCR 
reader = easyocr.Reader(['en'], gpu=False)

# ==========================================
# HÀM XỬ LÝ KẾT NỐI (PIPELINE)
# ==========================================
def process_license_plate(image_path):
    # Đọc ảnh gốc bằng OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc được file ảnh gốc. Hãy kiểm tra lại đường dẫn!")
        return

    # BƯỚC 1: Phát hiện vị trí biển số xe bằng YOLO11n từ ảnh gốc
    results = model_1.predict(source=img, conf=0.5, save=False, verbose=False)
    
    if len(results[0].boxes) == 0:
        print("Mô hình YOLO không tìm thấy biển số nào trong bức ảnh này.")
        return
        
    for idx, box in enumerate(results[0].boxes):
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
        
        # Tiến hành CẮT ẢNH theo tọa độ phát hiện được
        cropped_plate = img[y1:y2, x1:x2]
        
        # BƯỚC 2: Gọi EasyOCR nhận diện chuỗi ký tự trực tiếp từ ma trận ảnh cắt
        ocr_result = reader.readtext(cropped_plate)
        
        plate_text = ""
        
        # Trích xuất dữ liệu chuỗi từ kết quả của EasyOCR
        if ocr_result:
            for res in ocr_result:
                text = res[1]  # Lấy chuỗi chữ nhận diện được
                plate_text += " " + text
        
        # Định dạng chuẩn hóa chuỗi biển số (viết hoa, xóa dấu chấm, dấu gạch)
        plate_text = plate_text.strip().upper().replace("-", "").replace(".", "").replace(" ", "")
        
        print(f"--- Biển số thứ {idx + 1} ---")
        print(f"Tọa độ cắt từ ảnh gốc: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Kết quả nhận diện OCR chuỗi: {plate_text}")
        print("-" * 25)

# ==========================================
# CHẠY THỬ NGHIỆM (Truyền đường dẫn Ảnh Gốc vào đây)
# ==========================================
# LƯU Ý: Đường dẫn dưới đây phải là ảnh toàn cảnh chiếc xe (ảnh gốc chứa biển số chưa cắt)
process_license_plate("LAPTRINH PYTHON BTL /BTLPYTHON/SMP-2/train/images/123.jpg")