import cv2
import easyocr
from google.colab.patches import cv2_imshow

# ==========================================
# 1. KHỞI TẠO DUY NHẤT MODEL 2 (EASYOCR)
# ==========================================
reader = easyocr.Reader(['en'], gpu=False)

# ==========================================
# 2. ĐỌC THẲNG ẢNH ĐÃ CẮT VÀ DỊCH CHỮ
# ==========================================
# Đường dẫn chuẩn tới file ảnh BIỂN SỐ ĐÃ CẮT SẴN
IMAGE_PATH = "/content/output_cropped_plates/z7947458603191_60443c39ad13bf8c8c68beb0f1e6928a_jpg.rf.87c390ecb0dd5b5459b6c91b3e080b73_crop_0.jpg"

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Không tìm thấy hoặc không đọc được file ảnh.")
else:
    # Hiển thị ảnh biển số đang test
    print(" Ảnh biển số đầu vào:")
    cv2_imshow(img)

    # BỎ QUA BƯỚC YOLO - Đẩy thẳng ảnh vào EasyOCR để đọc chữ luôn
    ocr_result = reader.readtext(img)

    plate_text = ""
    if ocr_result:
        for res in ocr_result:
            plate_text += " " + res[1]

    # Chuẩn hóa văn bản đầu ra
    plate_text = plate_text.strip().upper().replace("-", "").replace(".", "").replace(" ", "")
    print(f"📝 Chuỗi ký tự dịch được: {plate_text}")
