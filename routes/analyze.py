from flask import Blueprint, request, jsonify, current_app
import os, uuid, traceback
import cv2
from inference import run_inference

analyze_bp = Blueprint("analyze", __name__)

@analyze_bp.route("/analyze", methods=["POST"])
def analyze():

    files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "No images provided"}), 400

    results = []

    try:

        for file in files:

            filename = f"{uuid.uuid4()}.jpg"

            input_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            output_path = os.path.join(current_app.config["RESULT_FOLDER"], filename)

            file.save(input_path)

            model = current_app.config["MODEL"]

            result_img, metrics = run_inference(input_path, model)

            cv2.imwrite(output_path, result_img)

            image_url = f"{current_app.config['BASE_URL']}/results/{filename}"

            results.append({
                "image_url": image_url,
                "metrics": metrics
            })

        return jsonify({"results": results})

    except Exception as e:

        traceback.print_exc()

        return jsonify({"error": str(e)}), 500