import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Generator
import helpers
from model import ImageClassifier
from torch import nn, Tensor
from PIL import Image, ImageFile.

# ==========================
# Config
# ==========================

DB_PATH = "src/database.db"
BATCH_SIZE = 32
IMAGE_FOLDER = Path("src/images")


# ==========================
# Logging
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ==========================
# Database
# ==========================

def get_connection():
    return sqlite3.connect(DB_PATH)


def fetch_new_items() -> List[Dict]:

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, image_path, category_id
        FROM returns
        WHERE status_id = 'NEW'
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "id": row[0],
            "image_path": row[1],
            "category_id": row[2]
        }
        for row in rows
    ]


def update_prediction(
    item_id: int,
    predicted_category: int,
    confidence: float,
    status: str
):
    """
    Update processed item
    """

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE returns
        SET predicted_category_id = ?,
            confidence = ?,
            status_id = ?
        WHERE id = ?
    """, (
        predicted_category,
        confidence,
        status,
        item_id
    ))

    conn.commit()
    conn.close()


# ==========================
# Batch Helpers
# ==========================

def chunk(items: List[Dict], batch_size: int) -> Generator[List[Dict], None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

# ==========================
# Model Inference
# ==========================

def load_model() -> ImageClassifier:
    return ImageClassifier()

def preprocess_image(model : ImageClassifier, path : str) -> Tensor:
    
    img = Image.open(path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return model.transform(img)

def predict_batch(model : ImageClassifier, batch : List[Dict]):
    
    processed_images : List[Dict] = []

    for item in batch:
        image_path = item[1]
        img = preprocess_image(model, image_path)
        processed_images.append({"id": item[0], "image": img, "category_id": item[2]})
        
    predictions = [model.forward(result[1]) for result in processed_images]




# ==========================
# Validation
# ==========================

def validate_prediction(
    submitted_category,
    predicted_category,
    confidence
):
    """
    Decide final status:
    OK / FLAGGED / UNCERTAIN
    """

    pass


# ==========================
# Processing Pipeline
# ==========================

def process_batch(model : ImageClassifier, batch : List[Dict]):

    logging.info(f"Processing batch size={len(batch)}")

    predictions = predict_batch(model, batch)

    for item, pred in zip(batch, predictions):

        status = validate_prediction(
            submitted_category=item["category_id"],
            predicted_category=pred["class"],
            confidence=pred["confidence"]
        )

        update_prediction(
            item_id=item["id"],
            predicted_category=pred["class"],
            confidence=pred["confidence"],
            status=status
        )


# ==========================
# Main Cron Entry
# ==========================

def run():

    logging.info("Nightly batch started")

    model = load_model()

    items = fetch_new_items()

    if not items:
        logging.info("No new items found")
        return

    logging.info(f"Found {len(items)} new items")

    for batch in chunk(items, BATCH_SIZE):
        process_batch(model, batch)

    logging.info("Nightly batch finished")


if __name__ == "__main__":
    run()