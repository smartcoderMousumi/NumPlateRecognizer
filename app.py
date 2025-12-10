import os
import cv2
from flask import Flask, render_template, request, url_for, redirect

from plateDetection import detect_plate
from ocrModule import read_plate_text
from database import init_db, insert_plate, get_all_logs, delete_log  # imported delete_log

BASE_DIR = os.path.dirname(__file__)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize DB (ensures table exists)
init_db()


@app.route("/", methods=["GET", "POST"])
def index():
    detected_text = None
    message = None
    plate_image_url = None

    if request.method == "POST":
        if "plate_image" not in request.files:
            message = "No file part in the form."
        else:
            file = request.files["plate_image"]

            if file.filename == "":
                message = "No file selected."
            else:
                # Save uploaded file
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                # Read with OpenCV
                image = cv2.imread(filepath)
                if image is None:
                    message = "Could not read the uploaded image."
                else:
                    # Detect plate
                    plate_img, coords = detect_plate(image)

                    if plate_img is None or coords is None:
                        message = "No plate detected in this image."
                    else:
                        # OCR
                        text, processed = read_plate_text(plate_img)

                        if text:
                            detected_text = text
                            message = "Plate detected successfully!"

                            # ✅ Log to database
                            insert_plate(detected_text, camera_location="Web Upload")
                        else:
                            message = "Plate region found, but OCR could not read the text clearly."

                        # Save cropped plate image for display
                        base_name = os.path.splitext(file.filename)[0]
                        plate_filename = f"{base_name}_plate.jpg"
                        plate_path = os.path.join(RESULT_FOLDER, plate_filename)
                        cv2.imwrite(plate_path, plate_img)

                        plate_image_url = url_for(
                            "static", filename=f"results/{plate_filename}"
                        )

    # ✅ Always fetch recent history (even on GET)
    history_rows = get_all_logs(limit=10)

    return render_template(
        "index.html",
        detected_text=detected_text,
        message=message,
        plate_image_url=plate_image_url,
        history=history_rows,
    )


@app.route("/delete/<int:row_id>", methods=["POST"])
def delete_entry(row_id):
    """Delete a log entry and redirect back to index."""
    try:
        delete_log(row_id)
    except Exception as e:
        # You may want to log the error in production
        print("Error deleting row:", e)
    return redirect(url_for("index"))

@app.route("/confirm", methods=["POST"])
def confirm_plate():
    """
    After detection, the UI will POST confirmed_text and (optionally) plate_image filename.
    We insert the confirmed value to DB and redirect back to home.
    """
    confirmed = request.form.get("confirmed_text", "").strip().upper()
    source = request.form.get("source", "Web Upload")
    if confirmed:
        # insert into DB
        insert_plate(confirmed, camera_location=source)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
