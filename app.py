from flask import Flask, request, jsonify, send_file, abort, make_response, url_for
from flask_cors import CORS
import os, uuid, traceback
import cv2
from ultralytics import YOLO
from inference import run_inference

HOST = "0.0.0.0"
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)  # позволяет GitHub Pages делать запросы
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    try:
        for file in files:
            filename = f"{uuid.uuid4()}.jpg"
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            output_path = os.path.join(app.config["RESULT_FOLDER"], filename)

            file.save(input_path)
            result_img, metrics = run_inference(input_path, model)
            cv2.imwrite(output_path, result_img)

            image_url = url_for("proxy_image", filename=filename, _external=True)
            results.append({"image_url": image_url, "metrics": metrics})

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/proxy_image/<filename>")
def proxy_image(filename):
    full_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    if not os.path.exists(full_path):
        return abort(404)

    ext = os.path.splitext(full_path)[1].lower()
    mimetype = "image/png" if ext == ".png" else "image/jpeg"

    response = make_response(send_file(full_path, mimetype=mimetype))
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print(f"Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)