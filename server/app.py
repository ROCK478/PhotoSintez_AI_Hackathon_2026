from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import cv2
from inference import run_inference
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

#model = YOLO("path_to_your_model.pt")  # путь к обученной модели

# Папки для хранения изображений
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    # Проверяем, есть ли файл
    if "image" not in request.files:
        return jsonify({"error": "Файл не отправлен"}), 400

    file = request.files["image"]
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Запускаем тестовую обработку
    result_img, metrics = run_inference(input_path)

    # Сохраняем результат
    output_path = os.path.join(RESULT_FOLDER, file.filename)
    cv2.imwrite(output_path, result_img)

    # Отправляем путь к изображению и метрики
    return jsonify({
        "result_image": output_path,
        "metrics": metrics
    })

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory("results", filename)

if __name__ == "__main__":
    app.run(debug=True)