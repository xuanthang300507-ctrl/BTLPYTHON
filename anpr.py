import os
import time
import requests
from data_detect import detect_and_crop_plate
from ocr_utils import recognize_plate_text

C_SERVER = "http://localhost:8080"
TARGET_IMAGE = "current_frame.bmp"
TRIGGER_FILE = "scan_trigger.flag"

def monitor():
    print("🚀 ANPR running...")

    last_trigger = 0

    while True:
        try:
            if os.path.exists(TRIGGER_FILE):
                t = os.path.getmtime(TRIGGER_FILE)

                if t != last_trigger:
                    last_trigger = t

                    if not os.path.exists(TARGET_IMAGE):
                        print("❌ No camera image")
                        continue

                    print("📸 Scanning...")

                    crop = detect_and_crop_plate(TARGET_IMAGE)

                    if crop is None:
                        print("❌ No plate detected")
                        continue

                    plate = recognize_plate_text(crop)

                    if not plate:
                        print("❌ OCR failed")
                        continue

                    print(f"✅ Plate: {plate}")

                    requests.post(
                        f"{C_SERVER}/api/scan",
                        data={"plate": plate},
                        timeout=5
                    )

        except Exception as e:
            print("ERROR:", e)

        time.sleep(0.3)

if __name__ == "__main__":
    monitor()