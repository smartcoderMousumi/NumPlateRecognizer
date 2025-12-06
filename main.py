import cv2
from plateDetection import detect_plate
from ocrModule import read_plate_text
from database import init_db, insert_plate


def process_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Error loading image")
        return

    plate_img, coords = detect_plate(image)
    if plate_img is None:
        print("No plate detected.")
        return

    text, processed = read_plate_text(plate_img)

    if text:
        print("Detected Plate:", text)
        insert_plate(text, camera_location="Test Camera")
    else:
        print("Plate region detected but OCR could not confidently read text.")

    # Draw rectangle if coords exist
    if coords is not None:
        x, y, w, h = coords
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Original with Plate Box", image)
    cv2.imshow("Cropped Plate", plate_img)
    cv2.imshow("Processed for OCR", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    init_db()
    process_image("images/car1.jpg")
