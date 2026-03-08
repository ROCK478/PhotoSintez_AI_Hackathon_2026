from flask import Blueprint, request, jsonify
from database.db import db
from database.models import User
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token

auth_bp = Blueprint("auth", __name__)

bcrypt = Bcrypt()

@auth_bp.route("/register", methods=["POST"])
def register():

    data = request.json

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error":"Missing fields"}), 400

    existing = User.query.filter_by(email=email).first()

    if existing:
        return jsonify({"error":"User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    user = User(
        username=username,
        email=email,
        password=hashed_password
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"status":"registered"})

@auth_bp.route("/login", methods=["POST"])
def login():

    data = request.json

    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"error":"User not found"}), 401

    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error":"Wrong password"}), 401

    token = create_access_token(identity=user.id)

    return jsonify({
        "access_token": token,
        "username": user.username
    })