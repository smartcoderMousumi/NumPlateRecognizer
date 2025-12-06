import cv2
import numpy as np
import pytesseract

# ✅ Make sure this path is correct on your machine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print("Using Tesseract from:", pytesseract.pytesseract.tesseract_cmd)


def preprocess_plate(plate_img):
    # 1) Grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 2) Denoise but keep edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # 3) Global Otsu threshold
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4) Make sure text is dark on white (Tesseract prefers)
    # If background is dark, invert
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # 5) Enlarge image to help Tesseract
    thresh = cv2.resize(
        thresh, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR
    )

    # 6) Slight morphological closing to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return thresh


def read_plate_text(plate_img):
    processed = preprocess_plate(plate_img)

    candidates = []

    # 🔁 Try a few Tesseract page segmentation modes
    for psm in [7, 8, 6]:
        config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        raw = pytesseract.image_to_string(processed, config=config)
        print(f"PSM {psm} raw OCR:", repr(raw))

        cleaned = "".join(ch for ch in raw.upper() if ch.isalnum())

        if cleaned:
            candidates.append(cleaned)

    if not candidates:
        return "", processed

    # For now, pick the longest non-empty candidate
    best = max(candidates, key=len)
    return best, processed
