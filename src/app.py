from flask import Flask, request, jsonify, render_template
import sqlite3
import os
import requests
from datetime import datetime
import base64

UPLOAD_FOLDER = "images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/dashboard")
def dashboard():
    conn = get_db_connection()
    returns = conn.execute(
        """
        SELECT 
            r.order_id,
            c.name AS category,
            s.name AS status
        FROM returns r
        JOIN categories c ON c.id = r.category_id
        JOIN statuses s ON s.id = r.status_id;
        """
        ).fetchall()
    conn.close()
    return render_template("returns.html", returns=returns)


@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    order_id = request.form["order_id"]
    category_id = int(request.form["category_id"])

    image_bytes = file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "order_id": order_id,
        "category_id": category_id,
        "image": image_base64
    }

    response = requests.post(
        "http://127.0.0.1:5000/api/upload",
        json=payload
    )

    return response.text, response.status_code


@app.route("/api/upload", methods=["POST"])
def upload():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = request.get_json()
    order_id = data["order_id"]
    category_id = data["category_id"]
    image_b64 = data["image"]
    unique_filename = f"{order_id}_{timestamp}"
    image_bytes = base64.b64decode(image_b64)
    image_path = f"{UPLOAD_FOLDER}/{unique_filename}.jpg"
    image_name = f"{unique_filename}.jpg"

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO returns
        (
            order_id,
            category_id,
            image_path
        )
        VALUES (?, ?, ?)
        """, (
        order_id,
        category_id,
        image_name
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "message": "uploaded",
        "path": image_name
    })

if __name__ == "__main__":
    app.run(debug=True)