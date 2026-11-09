
from pathlib import Path
import sys
import os
import uuid
import json
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    abort,
    redirect,
    url_for,
)

from werkzeug.utils import secure_filename


# ==========================================
# 1. MAIN PROJECT FOLDER
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from predictor import predict_image
from report_generator import generate_oa_report


# ==========================================
# 2. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "dev-only-change-this-secret"
)

UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# JSON file to store patient and prediction history
HISTORY_FILE = Path(__file__).resolve().parent / "history.json"

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ==========================================
# 3. HISTORY STORAGE FUNCTIONS
# ==========================================

def load_history():

    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)

            return records if isinstance(records, list) else []

    except (json.JSONDecodeError, OSError):
        app.logger.exception("Could not read history.json")
        return []


def save_history(record):

    records = load_history()

    # Add newest patient record at the beginning
    records.insert(0, record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4)


# ==========================================
# 4. CHECK IMAGE EXTENSION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================================
# 5. DISPLAY UPLOADED IMAGE
# ==========================================

@app.route("/uploaded-xray/<filename>")
def uploaded_xray(filename):

    filename = secure_filename(filename)

    if not filename:
        abort(404)

    return send_from_directory(UPLOAD_FOLDER, filename)


# ==========================================
# 6. DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    return render_template("dashboard.html")


# ==========================================
# 7. HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    records = load_history()

    # Count predictions for each KL grade
    grade_counts = {
        grade: 0 for grade in range(5)
    }

    for record in records:

        grade = record.get("grade")

        if grade in grade_counts:
            grade_counts[grade] += 1

    total_images = len(records)

    return render_template(
        "history.html",
        records=records,
        grade_counts=grade_counts,
        total_images=total_images
    )


# ==========================================
# 8. PERFORMANCE PAGE
# ==========================================

@app.route("/performance")
def performance():

    return render_template("performance.html")


# ==========================================
# 9. HOME PAGE AND IMAGE PROCESSING
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    # --------------------------------------
    # GET REQUEST
    # --------------------------------------

    if request.method == "GET":

        # Display prediction result once
        result = session.pop("oa_display_once", None)

        if result is None:

            # Clear previous PDF session information
            # Do not delete uploaded images needed by history
            session.pop("oa_report", None)

    # --------------------------------------
    # POST REQUEST
    # --------------------------------------

    else:

        # Clear previous session information
        session.pop("oa_report", None)
        session.pop("oa_display_once", None)

        # ----------------------------------
        # GET PATIENT DETAILS
        # ----------------------------------

        patient_name = request.form.get(
            "patient_name", ""
        ).strip()

        patient_id = request.form.get(
            "patient_id", ""
        ).strip()

        age_text = request.form.get(
            "age", ""
        ).strip()

        gender = request.form.get(
            "gender", ""
        ).strip()

        knee_side = request.form.get(
            "knee_side", ""
        ).strip()

        # ----------------------------------
        # GET X-RAY IMAGE
        # ----------------------------------

        file = request.files.get("xray")

        # ----------------------------------
        # VALIDATE PATIENT DETAILS
        # ----------------------------------

        if not patient_name or not patient_id or not age_text or not knee_side:

            error = "Please fill in all required patient details."

        elif len(patient_name) > 100 or len(patient_id) > 50:

            error = "Patient name or ID is too long."

        elif not age_text.isdigit() or not 1 <= int(age_text) <= 120:

            error = "Please enter a valid age between 1 and 120."

        elif gender not in ("", "Male", "Female", "Other"):

            error = "Please select a valid gender."

        elif knee_side not in ("Left", "Right"):

            error = "Please select Left or Right knee."

        elif not file or file.filename == "":

            error = "Please select a knee X-ray image."

        elif not allowed_file(file.filename):

            error = "Please upload a JPG, JPEG, or PNG image."

        else:

            original_filename = secure_filename(file.filename)

            if not original_filename:

                error = "Invalid filename. Please select another image."

            else:

                # Generate unique filename
                extension = original_filename.rsplit(".", 1)[1].lower()

                filename = f"{uuid.uuid4().hex}.{extension}"

                image_path = UPLOAD_FOLDER / filename

                try:

                    # ----------------------------------
                    # SAVE UPLOADED IMAGE
                    # ----------------------------------

                    file.save(image_path)

                    # ----------------------------------
                    # EXISTING AI PREDICTION
                    # ----------------------------------

                    prediction = predict_image(image_path)

                    grade = int(prediction["predicted_grade"])
                    confidence = float(prediction["confidence"])

                    # ----------------------------------
                    # CREATE RESULT
                    # ----------------------------------

                    result = {
                        "filename": filename,
                        "grade": grade,
                        "confidence": confidence,
                        "patient_name": patient_name,
                        "patient_id": patient_id,
                        "age": int(age_text),
                        "gender": gender,
                        "knee_side": knee_side,
                    }

                    # ----------------------------------
                    # SAVE PATIENT AND PREDICTION HISTORY
                    # ----------------------------------

                    history_record = {
                        "filename": filename,
                        "original_filename": original_filename,
                        "patient_name": patient_name,
                        "patient_id": patient_id,
                        "age": int(age_text),
                        "gender": gender,
                        "knee_side": knee_side,
                        "grade": grade,
                        "confidence": confidence,
                        "date": datetime.now().strftime(
                            "%d-%m-%Y %I:%M %p"
                        ),
                    }

                    save_history(history_record)

                    # ----------------------------------
                    # STORE RESULT FOR ONE-TIME DISPLAY
                    # ----------------------------------

                    session["oa_display_once"] = result

                    # Store result for PDF download
                    session["oa_report"] = result

                    # Redirect to home page
                    return redirect(url_for("home"))

                except Exception:

                    app.logger.exception("Image processing failed")

                    error = (
                        "Could not process this image. "
                        "Check the VS Code terminal for details."
                    )

                    # Delete image if processing fails
                    if image_path.exists():
                        image_path.unlink()

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# ==========================================
# 10. DOWNLOAD PDF REPORT
# ==========================================

@app.route("/download-report")
def download_report():

    report_data = session.get("oa_report")

    if not report_data:

        abort(
            404,
            description="Please process an X-ray before downloading a report."
        )

    filename = report_data["filename"]

    image_path = UPLOAD_FOLDER / filename

    if not image_path.is_file():

        abort(
            404,
            description="The uploaded X-ray image is no longer available."
        )

    try:

        pdf_buffer = generate_oa_report(
            image_path=image_path,
            filename=filename,
            grade=report_data["grade"],
            confidence=report_data["confidence"],
        )

        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="OA_Vision_AI_Report.pdf",
        )

    except Exception:

        app.logger.exception("PDF report generation failed")

        abort(
            500,
            description="Could not generate the PDF report. Check the terminal."
        )


# ==========================================
# 11. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
