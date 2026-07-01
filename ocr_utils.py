import easyocr
import re

reader = easyocr.Reader(['en'], gpu=False)

def normalize_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text if len(text) >= 5 else None

def recognize_plate_text(img):
    result = reader.readtext(img)

    if not result:
        return None

    text = " ".join([r[1] for r in result])
    return normalize_plate(text)