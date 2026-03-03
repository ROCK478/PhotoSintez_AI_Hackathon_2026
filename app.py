from flask import Flask, request, jsonify, send_from_directory, url_for, make_response
from flask_cors import CORS
import os
import cv2
import uuid
import traceback
from ultralytics import YOLO
from inference import run_inference


# =========================
# CONFIG
# =========================

HOST = "0.0.0.0"
PORT = 8000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


print("BASE_DIR:", BASE_DIR)
print("UPLOAD_FOLDER:", UPLOAD_FOLDER)
print("RESULT_FOLDER:", RESULT_FOLDER)


# =========================
# INIT APP
# =========================

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

CORS(app)


# =========================
# LOAD MODEL
# =========================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")


# =========================
# ANALYZE ENDPOINT
# =========================

@app.route("/analyze", methods=["POST"])
def analyze():

    files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []

    try:
        host = request.headers.get("Host")

        for file in files:

            filename = f"{uuid.uuid4()}.jpg"

            input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            output_path = os.path.join(app.config["RESULT_FOLDER"], filename)

            file.save(input_path)

            result_img, metrics = run_inference(input_path, model)

            cv2.imwrite(output_path, result_img)

            if "127.0.0.1" in host or "localhost" in host:
                image_url = f"http://{host}/image/{filename}"
            else:
                image_url = f"https://{host}/image/{filename}"

            results.append({
                "image_url": image_url,
                "metrics": metrics
            })

        return jsonify({
            "results": results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# =========================
# IMAGE ENDPOINT (NO NGROK WARNING)
# =========================

@app.route("/image/<filename>")
def get_image(filename):

    full_path = os.path.join(app.config["RESULT_FOLDER"], filename)

    print("Requested image:", full_path)

    if not os.path.exists(full_path):
        print("FILE NOT FOUND")
        return jsonify({"error": "File not found"}), 404

    response = make_response(
        send_from_directory(
            app.config["RESULT_FOLDER"],
            filename
        )
    )

    response.headers["Content-Type"] = "image/jpeg"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["ngrok-skip-browser-warning"] = "true"

    return response


# =========================
# OPTIONAL: direct results access
# =========================

@app.route("/results/<filename>")
def get_result(filename):

    full_path = os.path.join(app.config["RESULT_FOLDER"], filename)

    if not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(
        app.config["RESULT_FOLDER"],
        filename
    )


# =========================
# HEALTH CHECK
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# =========================
# START SERVER
# =========================

if __name__ == "__main__":

    print(f"Starting server on {HOST}:{PORT}")

    app.run(
        host=HOST,
        port=PORT,
        debug=False
    )