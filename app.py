import os
from flask import Flask, render_template, request
import cv2

from plateDetection import detect_plate
from ocrModule import read_plate_text  # uses your existing Tesseract setup

# ---- BASIC FLASK SETUP ----
app = Flask(__name__)

# Folder to save uploads (make sure this exists)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    detected_text = None
    message = None

    if request.method == "POST":
        if "plate_image" not in request.files:
            message = "No file part in the form."
            return render_template("index.html", detected_text=detected_text, message=message)

        file = request.files["plate_image"]

        if file.filename == "":
            message = "No file selected."
            return render_template("index.html", detected_text=detected_text, message=message)

        # Save uploaded file
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Read with OpenCV
        image = cv2.imread(filepath)
        if image is None:
            message = "Could not read the uploaded image."
            return render_template("index.html", detected_text=detected_text, message=message)

        # Run your existing detection + OCR
        plate_img, coords = detect_plate(image)

        if plate_img is None:
            message = "No plate detected in this image."
            return render_template("index.html", detected_text=detected_text, message=message)

        text, processed = read_plate_text(plate_img)

        if text:
            detected_text = text
            message = "Plate detected successfully!"
        else:
            message = "Plate region found, but OCR could not read text clearly."

    return render_template("index.html", detected_text=detected_text, message=message)


if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True)
