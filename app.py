from flask import Flask, send_file
from flask_cors import CORS
import os
from ultralytics import YOLO
from database.db import db
from routes.analyze import analyze_bp
from routes.auth import auth_bp
from flask_jwt_extended import JWTManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")

app = Flask(__name__, static_folder="frontend", static_url_path="")
app.add_url_rule('/results/<path:filename>',
                 endpoint='results',
                 view_func=lambda filename: send_file(os.path.join(RESULT_FOLDER, filename)))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["JWT_SECRET_KEY"] = "super-secret-key"
app.config["BASE_URL"] = "http://127.0.0.1:8000"
CORS(app)
db.init_app(app)
jwt = JWTManager(app)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully")

app.config["MODEL"] = model

app.register_blueprint(analyze_bp)
app.register_blueprint(auth_bp)

@app.route("/")
def index():
    return send_file("frontend/login.html")

@app.route("/main")
def login_page():
    return send_file("frontend/index.html")

@app.route("/pricing")
def pricing_page():
    return send_file("frontend/pricing.html")

@app.route("/profile")
def profile_page():
    return send_file("frontend/profile.html")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=8000)