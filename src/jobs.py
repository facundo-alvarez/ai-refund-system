import os
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Generator
from model import ImageClassifier
from db import DatabaseInitializer
import torch
from torch import Tensor
from PIL import Image
from datetime import datetime

# Configuration constants

DB_PATH = os.path.join("instance", "database.db")
BATCH_SIZE = 32
IMAGE_FOLDER = "images"
MODEL_WEIGHTS = "models/best_model.pth"

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def __get_connection():
    """
    Opens the connection to the database
    """

    return sqlite3.connect(DB_PATH)


def __fetch_new_items() -> List[Dict]:
    """
    Gets new items from the database that with the flag 'New'
    """

    conn = __get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            r.id,
            r.image_path,
            r.category_id,
            s.name
        FROM returns r
        JOIN statuses s
        ON r.status_id = s.id
        WHERE s.name = 'New'
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


def __update_prediction(
    item_id: int,
    predicted_category: int,
    confidence: float,
    status: str
):
    """
    Update processed item

    `status` is the status *name* (e.g. "Flagged"); it is resolved to the
    matching statuses.id before the write since status_id is a foreign key.
    """

    conn = __get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM statuses WHERE name = ?", (status,))
    status_id = cursor.fetchone()[0]

    cursor.execute("""
        UPDATE returns
        SET predicted_category_id = ?,
            confidence = ?,
            status_id = ?
        WHERE id = ?
    """, (
        predicted_category + 1,
        confidence,
        status_id,
        item_id
    ))

    conn.commit()
    conn.close()


def __chunk(items: List[Dict], batch_size: int) -> Generator[List[Dict], None, None]:
    """ A helper function that takes a batch of items and returns a generator """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def __load_model(weights:str, device="cpu") -> ImageClassifier:
    """
    Loads the image classifier model and uses the stored weights for the model
    """
    model = ImageClassifier()
    checkpoint = torch.load(weights, map_location=device)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"]) 
    model.eval()
    return model


def __preprocess_image(model : ImageClassifier, path : str) -> Tensor:
    """
    The actual processing of the image
    """

    img = Image.open(Path(IMAGE_FOLDER, path))

    if img.mode != "RGB":
        img = img.convert("RGB")

    return model.transform(img)


def __predict_batch(model : ImageClassifier, batch : List[Dict], device='cpu') -> List[Dict]:
    """
    Takes a batch of results, process them and run predictions that are returned as a dictionary
    """
    predictions = []

    for item in batch:

        img = __preprocess_image(model, item["image_path"])
        img = img.to(device)
        pred = model(img)

        probs = pred.softmax(dim=1)
        predicted_class = probs.argmax(dim=1).item()
        confidence = probs.max().item()

        predictions.append({
            "id": item["id"],
            "class": predicted_class,
            "confidence": confidence,
            "category_id": item["category_id"]
        })

    return predictions


def __validate_prediction(submitted_category : int, predicted_category : int, confidence : float) -> str:
    """ 
    Validation logic of the image classifier output 
    """

    if confidence < 0.7 or submitted_category != predicted_category + 1:
        return "Flagged"
    
    return "Processed"


def __process_batch(model : ImageClassifier, batch : List[Dict], device = 'cpu'):
    """
    Process the batch of data using the provided image classifier and 
    validates against the database and update the record.
    """

    logging.info(f"Processing batch size={len(batch)}")

    predictions = __predict_batch(model, batch, device)

    for item, pred in zip(batch, predictions):

        predicted = pred["class"] + 1
        confidence = pred["confidence"]

        status = __validate_prediction(
            submitted_category=item["category_id"],
            predicted_category=predicted,
            confidence=confidence
        )

        __update_prediction(
            item_id=item["id"],
            predicted_category=predicted,
            confidence=confidence,
            status=status
        )


def __run():
    """
    Main cron job entry point
    """

    logging.info("Nightly batch started")

    db_init = DatabaseInitializer(DB_PATH)
    db_init.setup_database()
    db_init.close()

    if not os.path.exists(MODEL_WEIGHTS):
        logging.warning(
            "No model weights found at '%s' - skipping batch run. "
            "Train a model first (see README).", MODEL_WEIGHTS
        )
        return

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model = __load_model(MODEL_WEIGHTS, device)

    items = __fetch_new_items()

    if not items:
        logging.info("No new items found")
    else:
        logging.info(f"Found {len(items)} new items")

        for batch in __chunk(items, BATCH_SIZE):
            __process_batch(model, batch, device)

        logging.info("Nightly batch finished")

    try:
        with open("/var/log/cron.log", "a") as f:
            f.write(f"Run on: {datetime.now()}\n")
    except OSError:
        # /var/log/cron.log only exists inside the worker Docker image.
        pass


if __name__ == "__main__":
    __run()