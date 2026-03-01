from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import uuid
import traceback
from ultralytics import YOLO
from inference import run_inference

app = Flask(__name__)
CORS(app)
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        print("No image in request")
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    filename = str(uuid.uuid4()) + ".jpg"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(RESULT_FOLDER, filename)
    file.save(input_path)

    try:

        result_img, metrics = run_inference(input_path, model)
        success = cv2.imwrite(output_path, result_img)
        if not success:
            raise Exception("cv2.imwrite failed")

        url = f"http://localhost:5000/results/{filename}"
        print("METRICS SENT TO FRONTEND:", metrics)
        return jsonify({
        "image_url": f"http://127.0.0.1:5000/results/{filename}",
        "metrics": metrics
        })

    except Exception as e:

        print("\n=== REQUEST ERROR ===")
        traceback.print_exc()
        print("=====================\n")

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)