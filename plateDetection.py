import cv2
import os

# --------- LOAD CASCADE SAFELY ----------

# Try to use OpenCV's built-in cascade first
opencv_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")

if os.path.exists(opencv_cascade_path):
    cascade_path = opencv_cascade_path
else:
    # Fallback to local file in your project folder
    # Make sure 'haarcascadePlate.xml' is a VALID cascade file
    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascadePlate.xml")

print("Using cascade:", cascade_path)

plate_cascade = cv2.CascadeClassifier(cascade_path)

if plate_cascade.empty():
    raise RuntimeError(f"Error: Could not load cascade file from: {cascade_path}")


# --------- DETECTION FUNCTION ----------

def detect_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 20)
    )

    if len(plates) == 0:
        return None, None

    x, y, w, h = sorted(plates, key=lambda r: r[2] * r[3], reverse=True)[0]

    # 🧩 Add small padding around plate
    pad = 5
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, image.shape[1])
    y2 = min(y + h + pad, image.shape[0])

    plate_img = image[y1:y2, x1:x2]
    return plate_img, (x1, y1, x2 - x1, y2 - y1)
