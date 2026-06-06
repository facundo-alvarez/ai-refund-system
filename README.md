# AI Refund System

An end-to-end image-classification system for processing product returns. Users upload a
photo of a returned item; a fine-tuned CNN classifies it into a product category, and the
result is stored alongside the return record.

## Features

- **Transfer learning** — a pretrained **ResNet152** is fine-tuned to classify items into
  10 fashion categories (Dress, Hat, Longsleeve, Outwear, Pants, Shirt, Shoes, Shorts,
  Skirt, T-Shirt).
- **Robust training pipeline**:
  - Seeded 70/15/15 train/validation/test split
  - **Class-imbalance handling** with a `WeightedRandomSampler` (inverse-frequency weights)
  - Adam with weight decay, **early stopping**, and **checkpoint save/resume**
- **Two-tier service** — a Flask web layer forwards uploads to an internal API endpoint
  that decodes, stores the image, and writes metadata to the database.
- **Normalized data model** — SQLite schema with `returns`, `categories`, and `statuses`
  tables joined for the dashboard view.

## Tech stack

- Python, PyTorch / torchvision (ResNet152)
- Flask
- SQLite
- Pillow

## Getting started

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python src/app.py
```

Then open `http://127.0.0.1:5000/upload` to submit a return, or `/dashboard` to view
stored returns.

## Training

Place the dataset under `data/` (images plus a `filter_data.csv` metadata file with a
`label` column), then run the training script. The best-performing model is checkpointed
to `models/best_model.pth` and final test accuracy is reported.

## Project structure

```
src/app.py     # Flask web + internal API + dashboard
src/model.py   # ResNet152-based ImageClassifier
src/train.py   # Training loop (sampler, early stopping, checkpointing)
```

> Note: A portfolio project demonstrating an end-to-end ML workflow — train, serve, persist.
