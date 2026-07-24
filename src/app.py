from flask import Flask, request, jsonify, render_template
import sqlite3
import os
import requests
from datetime import datetime
import base64
from db import DatabaseInitializer

UPLOAD_FOLDER = "images"
DB_PATH = os.path.join("instance", "database.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure the schema (and seed data) exists before the first request.
# Runs both under `python app.py` and `flask run` since it happens at import time.
_db_init = DatabaseInitializer(DB_PATH)
_db_init.setup_database()
_db_init.close()

app = Flask(__name__)

def __get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/dashboard")
def dashboard():
    """
    Render the dashboard page.

    Returns:
        str: Rendered HTML for the dashboard.
    """
    conn = __get_db_connection()
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
    """
    Render the returns form page.

    Returns:
        str: Rendered HTML for the dashboard.
    """
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Handle file upload requests.

    Receives an uploaded file, order and category from the form,
    converts the file to Base64, and forwards the data to the internal
    upload API endpoint.

    Returns:
        Response: Rendered upload result page containing:
            - success status
            - HTTP status code
            - API response text or error message
            - order ID
    """
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

    try:
        response = requests.post(
            "http://127.0.0.1:5000/api/upload",
            json=payload,
            timeout=10
        )

        success = response.status_code == 200

        return render_template(
            "upload_result.html",
            success=success,
            status_code=response.status_code,
            response_text=response.text,
            order_id=order_id
        )

    except Exception as e:
        return render_template(
            "upload_result.html",
            success=False,
            status_code=500,
            response_text=str(e),
            order_id=order_id
        )

@app.route("/api/upload", methods=["POST"])
def upload():
    """
    Process image upload API requests.

    Accepts JSON payload containing an order, category and a Base64-encoded image.
    Decodes and saves the image to disk, then stores metadata in the database.

    Expected JSON:
        {
            "order_id": str,
            "category_id": int,
            "image": str (base64 encoded)
        }

    Returns:
        JSON: Confirmation message and saved image filename.
    """

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

    conn = __get_db_connection()
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