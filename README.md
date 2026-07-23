# Wellbore Geology Prediction

End-to-end system for predicting true vertical thickness (TVT) along horizontal
wellbores from well log data, using a stacked LightGBM/CatBoost ensemble trained 
with data from ROGII - Wellbore Geology Prediction on Kaggle. Consists of a FastAPI 
backend (feature engineering + model inference) and a React/TypeScript frontend 
(upload a well CSV, view predictions, download results).

## Project structure

```
geology-competition/
├── app.py                  # FastAPI app: /health, /predict, /predict_batch
├── config.yaml             # models_dir, model_profile, S3 config, CORS, host/port
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── src/
│   ├── features.py         # build_well(): feature engineering pipeline
│   ├── model.py             # WellborePredictor: loads models, runs inference
│   └── models/              # local cache for downloaded model artifacts
│       ├── full/            # 3-seed LightGBM + 3-seed CatBoost ensemble
│       └── demo/             # lightweight 1-seed version (for free-tier hosting)
├── notebooks/
│   └── original_version.ipynb   # training notebook (source of truth for features/models)
└── frontend-ts/              # React + TypeScript frontend (separate deployable)
    ├── src/App.tsx
    └── ...
```

## Backend

FastAPI service that turns a raw horizontal-well CSV into TVT predictions.

### Local setup

```bash
pip install -r requirements.txt   # includes boto3, fastapi, uvicorn, python-multipart, lightgbm, catboost...
uvicorn app:app --reload
```
Runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### Configuration (`config.yaml`)

| Key | Purpose |
|---|---|
| `model_profile` | `full` or `demo` — which model set to load (see below). Overridable via `MODEL_PROFILE` env var without touching this file. |
| `model_profiles` | Per-profile definitions: which `.pkl` files to load and which S3 zip to pull them from if missing locally. |
| `models_s3_bucket` | S3 bucket the model artifacts live in. |
| `models_dir` | Optional override for where models are cached locally (defaults to `src/models/<profile>/`). |
| `type_well_csv` | Default type-well reference curve, used when a request doesn't upload its own. |
| `cors_origins` | Allowed origins for the frontend to call this API. `["*"]` by default. |
| `host` / `port` / `reload` | Passed to `uvicorn.run()` when running via `python app.py` directly. |

### Model profiles (full vs. demo)

The real ensemble (3 LightGBM + 3 CatBoost seeds) is too large for free-tier
hosting. `model.py` supports swapping in a lighter `demo` profile (1 CatBoost) 
via config — same code, same Docker image, different deployment:

```yaml
model_profile: demo   # or set MODEL_PROFILE=demo as an env var on the host instead
```

On first run, if the configured profile's `.pkl` files aren't present locally,
`WellborePredictor` automatically downloads and extracts them from S3
(`models.zip` for `full`, `models_demo.zip` for `demo`) — no manual
`download_models.py` step required. Requires AWS credentials
(`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`) available in the environment,
unless the bucket is public.

### API

- `GET /health` → `{status, model_profile, lgb_seeds, cat_seeds}`
- `POST /predict` (`multipart/form-data`)
  - `file` — horizontal well CSV (required)
  - `well_id` — optional, else derived from the filename
  - `type_well_file` — optional type-well reference CSV; falls back to `config.yaml`'s default
  - → `{well, n_points, predictions: [{id, md, pred}, ...]}`
- `POST /predict_batch` (`multipart/form-data`)
  - `files` — one or more well CSVs
  - `type_well_file` — optional, shared across all wells
  - → `{results: [...], errors: [...]}`

### Docker

```bash
docker build -t wellbore-geology-api:v1 .
docker run -d -p 8000:8000 --name geology-service \
  -e MODEL_PROFILE=full \
  -e AWS_ACCESS_KEY_ID=xxx \
  -e AWS_SECRET_ACCESS_KEY=xxx \
  wellbore-geology-api:v1
docker logs -f geology-service   # watch startup + request logs
```

## Frontend (`frontend-ts/`)

Minimal React + TypeScript (Vite) app: upload a CSV, run a prediction, view
results in a table, download them as CSV.

```bash
cd frontend-ts
npm install
cp .env.example .env   # set VITE_API_BASE_URL to your backend's URL
npm run dev            # http://localhost:5173
```

Production build: `npm run build` → static files in `dist/`, deployable to
Vercel/Netlify/any static host. Set `VITE_API_BASE_URL` as an environment
variable on the hosting provider so the built site points at the right
backend (same pattern as `MODEL_PROFILE` on the backend).

## Deploying both together

1. **Backend (Render)** — deploy from the Dockerfile. Environment variables:
   `MODEL_PROFILE` (`full` or `demo`), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.
   For a free-tier demo instance, set `MODEL_PROFILE=demo`.
2. **Frontend (Vercel)** — deploy `frontend-ts/`. Environment variable:
   `VITE_API_BASE_URL` = your Render backend's public URL.
3. Update the backend's `cors_origins` in `config.yaml` to the frontend's
   deployed URL once you have it (or leave `["*"]` for a demo).