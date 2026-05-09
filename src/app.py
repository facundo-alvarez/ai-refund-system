from flask import Flask, request, jsonify, render_template
import sqlite3
import os
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
        SELECT returns.*, categories.name AS category_name
        FROM returns
        JOIN categories ON returns.category_id = categories.id
        """
        ).fetchall()
    conn.close()
    return render_template("returns.html", returns=returns)


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
        image_path
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "message": "uploaded",
        "path": image_path
    })

if __name__ == "__main__":
    app.run(debug=True)