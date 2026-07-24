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
