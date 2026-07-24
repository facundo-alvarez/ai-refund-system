<<<<<<< HEAD
# RefundApp

A small end-to-end ML application for triaging clothing returns. A customer
(or agent) uploads a photo of a returned item together with the order number
and the category they claim it belongs to. A background worker runs an image
classifier (a fine-tuned ResNet152) against new uploads and flags any return
where the photo doesn't match the claimed category, or where the model isn't
confident, for manual review.

Built as a "model to production" exercise: dataset → training → serving →
a minimal ops dashboard.

## How it works

```
 ┌─────────────┐   upload photo    ┌──────────────┐        ┌──────────────┐
 │   Browser   │ ────────────────▶ │  Flask app   │──────▶ │   SQLite DB  │
 │ /upload UI  │                   │  (web)       │        │ (returns,    │
 └─────────────┘                   └──────────────┘        │ categories,  │
                                                              │ statuses)    │
 ┌─────────────┐   GET /dashboard  ┌──────────────┐        └──────┬───────┘
 │   Browser   │ ────────────────▶ │  Flask app   │               │
 └─────────────┘                   └──────────────┘               │
                                                                    │
                                     ┌──────────────┐               │
                                     │   Worker     │───────────────┘
                                     │ (cron, every │  reads "New" returns,
                                     │  minute)     │  classifies the photo,
                                     └──────────────┘  writes back a status
```

- **`web`** — a Flask app that serves an upload form, an internal JSON API
  (`POST /api/upload`) that saves the image and a `returns` row with status
  `New`, and a `/dashboard` page listing all returns with their category and
  status.
- **`worker`** — a cron job (every minute) that loads the trained model,
  picks up rows with status `New`, classifies the uploaded photo, and marks
  the row `Processed` (prediction agrees with the claimed category, with
  ≥70% confidence) or `Flagged` (otherwise).
- **model** — a ResNet152 (ImageNet-pretrained) fine-tuned to classify 10
  clothing categories: Dress, Hat, Longsleeve, Outwear, Pants, Shirt, Shoes,
  Shorts, Skirt, T-Shirt.

## Repository layout

```
src/
  app.py          Flask web app (upload UI, upload API, dashboard)
  db.py           SQLite schema creation + seed data
  jobs.py         Worker: nightly/periodic classification batch job
  model.py        ImageClassifier (ResNet152) definition
  dataset.py      PyTorch Dataset for training
  train.py        Training script
  templates/      Jinja templates for the web UI
  data_exploration.ipynb   Exploratory notebook over the training data
  instance/       SQLite database file (created automatically, gitignored)
  images/         Uploaded return photos (created automatically, gitignored)
  models/         Trained model weights (gitignored, see "Training a model")
data/             Training dataset: images + label CSVs (gitignored, see below)
docker/           Dockerfiles + compose.yaml for the web/worker containers
requirements.txt        Runtime deps for the web app (and worker base image)
requirements-train.txt  Extra deps needed to train locally or run the worker
                         outside Docker (torch, torchvision, pandas, numpy)
```

## Quickstart (web app only, no ML)

You can run the upload/dashboard UI without a trained model — the worker
will simply skip classification until model weights are present.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cd src
python app.py    # http://127.0.0.1:5000
```

The SQLite database and its schema are created automatically on first run
at `src/instance/database.db`. Open `http://127.0.0.1:5000/upload` to submit
a return, and `http://127.0.0.1:5000/dashboard` to see it listed.

## Running the worker locally

The worker needs `torch`/`torchvision` in addition to the base requirements:

```bash
pip install -r requirements.txt -r requirements-train.txt
```

(For a CUDA build, install `torch`/`torchvision` via the selector at
https://pytorch.org/get-started/locally/ instead of the pinned CPU-friendly
versions in `requirements-train.txt`.)

```bash
cd src
python jobs.py
```

This does one classification pass over all `New` returns and exits. Without
`src/models/best_model.pth` present, it logs a warning and exits without
doing anything — train a model first (see below) or drop in your own weights
at that path.

## Training a model

The training data isn't included in this repository (it's several thousand
images). It's derived from the public
[Clothing dataset (full, high resolution)](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)
on Kaggle.

1. Download the dataset and place it so you have:
   - `data/images/<image-id>.jpg` for each image
   - `data/filter_data.csv` with `image,label` columns (a filtered subset of
     the dataset's `images.csv`, dropping rows labeled `Not sure`/`Skip`)
2. Install training deps: `pip install -r requirements.txt -r requirements-train.txt`
3. From `src/`, run:
   ```bash
   python train.py
   ```
   This fine-tunes a ResNet152 with a weighted sampler (to counter class
   imbalance) and early stopping, and saves the best checkpoint to
   `src/models/best_model.pth` — the path `jobs.py` expects.

## Running with Docker

```bash
cd docker
docker compose up --build
```

This builds and starts two containers:

- `web` — the Flask app, published on `http://localhost:5000`
- `worker` — runs `jobs.py` once a minute via cron (built on the official
  `pytorch/pytorch` CUDA image, so no separate `requirements-train.txt`
  install is needed here)

Both containers share `src/instance` (the SQLite DB), `src/images`
(uploaded photos) and, for the worker, `src/models` (trained weights) as
bind-mounted volumes, so data survives rebuilds and is visible to both
services. Bring your own `src/models/best_model.pth` (see "Training a
model") for the worker to actually classify anything — without it, it logs
a warning every run and does nothing.

## API

`POST /api/upload` — internal JSON endpoint used by the upload form:

```json
{
  "order_id": "A1234",
  "category_id": 3,
  "image": "<base64-encoded JPEG>"
}
```

`category_id` refers to the `categories` table seeded by `db.py` (1=Dress,
2=Hat, 3=Longsleeve, 4=Outwear, 5=Pants, 6=Shirt, 7=Shoes, 8=Shorts,
9=Skirt, 10=T-Shirt). See `src/requests.http` for a ready-to-run example
(usable with the VS Code/JetBrains HTTP client extensions).

## Known limitations

- The Flask dev server (`debug=True` / `flask run --debug`) is used for both
  local development and the Docker image. It's convenient for this project's
  scope but is not hardened for a public deployment — put it behind a real
  WSGI server (gunicorn, etc.) and disable debug mode before exposing it
  beyond localhost.
- No authentication on the upload endpoint or dashboard.
- Single-node SQLite storage; fine for a demo, not for concurrent production
  load.
=======
# AI Refund System

An end-to-end machine-learning prototype system for automating product-return classification. Customers
submit a photo of a returned item through a web interface; a fine-tuned CNN classifies
the item, and a scheduled worker validates each prediction and routes it either to
automatic processing or to human review.

## Architecture

The system runs as two containerized services that share a SQLite database and an
image store:

- **web** (`docker/Dockerfile.web`) — a Flask application that accepts return
  submissions and serves a dashboard. New submissions are stored with status `New`.
- **worker** (`docker/Dockerfile.worker`) — a batch classifier that runs on a cron
  schedule, loads the trained model, and processes all pending returns.

![Infraestructure](infraestructure.jpg)

## Features

### Model
- **Transfer learning** — a pretrained **ResNet152** is fine-tuned to classify items
  into 10 clothing categories (Dress, Hat, Longsleeve, Outwear, Pants, Shirt, Shoes,
  Shorts, Skirt, T-Shirt).
- **Training pipeline** (`src/train.py`):
  - Seeded 70/15/15 train/validation/test split
  - **Class-imbalance handling** with a `WeightedRandomSampler` (inverse-frequency weights)
  - Adam with weight decay, **early stopping**, and **checkpoint save/resume**

### Web service
- **Two-tier design** — a frontend route forwards uploads to an internal API endpoint
  that decodes the image, stores it, and writes the return record to the database.
- **Dashboard** for reviewing stored returns and their predicted categories and statuses.

### Scheduled worker (batch classification)
- Runs `src/jobs.py` on a cron schedule inside the worker container.
- Fetches all returns with status `New`, loads the trained classifier
  (`models/best_model.pth`), and runs inference in **batches of 32** (GPU-accelerated
  when available).
- **Confidence- and consistency-based routing** (human-in-the-loop):
  - **Flagged** — prediction confidence below 0.7, *or* the predicted category does not
    match the category the customer submitted → sent for human review.
  - **Processed** — confident prediction that agrees with the submitted category.
- Writes the predicted category, confidence score, and updated status back to the database.

## Tech stack

- Python, PyTorch / torchvision (ResNet152)
- Flask
- SQLite
- Pillow
- Docker (two services: web + cron worker)

## Data model

A normalized SQLite schema, initialized and seeded by `src/db.py`:

- **categories** — the 10 clothing categories.
- **statuses** — `New`, `Processed`, `Flagged`.
- **returns** — return records: `order_id`, submitted `category_id`,
  `status_id`, `predicted_category_id`, `confidence`, and `image_path`, with foreign
  keys to `categories` and `statuses`.

## Getting started

### Local

```bash
pip install -r requirements.txt

# Initialize the database (creates tables + seed data)
python src/db.py

# Run the web app
python src/app.py
```

Open `http://127.0.0.1:5000/upload` to submit a return, or `/dashboard` to view stored
returns.

### Docker

```bash
# Web API
docker build -f docker/Dockerfile.web -t ai-refund-web .
docker run -d -p 5000:5000 -v "$(pwd)/data:/src/data" ai-refund-web

# Worker (scheduled batch classifier)
docker build -f docker/Dockerfile.worker -t ai-refund-worker .
docker run -d -v "$(pwd)/data:/src/data" ai-refund-worker
```

Both containers share the database and image folder through the mounted volume. The
worker's schedule is defined in `docker/Dockerfile.worker` (`/etc/cron.d/jobs`).

## Training

Place the dataset under `data/` (images plus a metadata CSV with a `label` column),
then run the training script. The best-performing checkpoint is saved to
`models/best_model.pth` and final test accuracy is reported.

## Project structure

```
src/app.py     # Flask web + internal API + dashboard
src/model.py   # ResNet152-based ImageClassifier
src/train.py   # Training loop (weighted sampler, early stopping, checkpointing)
src/jobs.py    # Scheduled batch classifier + confidence-based routing
src/db.py      # SQLite schema initialization + seed data
docker/        # Dockerfile.web and Dockerfile.worker
```

> Note: A portfolio project demonstrating an end-to-end ML workflow — train, serve,
> schedule, and route with a human-in-the-loop review step.
>>>>>>> 82e6e5d75ee2bea3bc68812e2eb74a4588f154cb
