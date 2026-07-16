"""
FastAPI REST API for the wellbore TVT predictor.

Endpoints
---------
GET  /health
    Liveness check.

POST /predict   (multipart/form-data)
    file:            the horizontal well CSV to predict on (required)
    well_id:         optional override, else derived from the uploaded filename
    type_well_file:  optional type-well reference CSV (TVT, GR columns);
                      if omitted, falls back to config.yaml's type_well_csv
    -> { "well": ..., "n_points": ..., "predictions": [ {id, md, pred}, ... ] }

POST /predict_batch   (multipart/form-data)
    files:           one or more horizontal well CSVs (required, repeat the field)
    type_well_file:  optional type-well reference CSV, shared across all wells
    -> { "results": [ <same shape as /predict per well>, ... ], "errors": [...] }

Run with:  uvicorn app:app --host 0.0.0.0 --port 8000
(host/port/reload can also be read from config.yaml if you run this file directly)

Requires `python-multipart` (needed by FastAPI/Starlette to parse file uploads):
    pip install python-multipart

Notes
-----
- The two spatial imputers (FormationPlaneKNN, DenseANCCImputer) are NOT rebuilt
  here -- they were already fit at training time and are loaded from disk inside
  WellborePredictor (self.FI / self.DI), via FI_knn.pkl / DI_imputer.pkl. We just
  reuse predictor.FI / predictor.DI when calling build_well().
- "tw" (type well curve: TVT vs GR reference log) can be uploaded per-request via
  type_well_file, or falls back to a CSV configured in config.yaml (key:
  type_well_csv), loaded once at startup.
- Feature building + model inference are CPU-bound / synchronous (pandas, numba,
  lightgbm, catboost), so request handlers offload them to a thread pool via
  run_in_threadpool instead of blocking the event loop. File uploads are read
  into memory (bytes) on the async side first, then handed to the threadpool as
  plain DataFrames -- no disk paths involved.
"""

import io
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.features import build_well, FormationPlaneKNN, DenseANCCImputer
from src.model import WellborePredictor

# model.py already does this patch, but it only patches whatever module happens
# to be sys.modules['__main__'] at that moment. If app.py ever gets imported
# twice under different module names (e.g. via `uvicorn.run("app:app", ...)`
# while also being executed directly), the two copies can disagree about which
# object __main__ is. Re-doing the patch here, right before we unpickle
# anything, makes sure the classes are always reachable via __main__ in
# *this* process no matter how we got here.
sys.modules["__main__"].FormationPlaneKNN = FormationPlaneKNN
sys.modules["__main__"].DenseANCCImputer = DenseANCCImputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wellbore-app")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config():
    defaults = {
        "models_dir": None,        # None -> WellborePredictor default (src/models)
        "type_well_csv": None,     # default type-well reference curve (TVT, GR columns)
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        defaults.update(user_cfg)
    else:
        log.warning("config.yaml not found at %s, using defaults", CONFIG_PATH)
    return defaults


CONFIG = load_config()

app = FastAPI(title="Wellbore TVT Predictor", version="2.0.0")

log.info("Loading WellborePredictor (profile=%s, models_dir=%s)...",
          CONFIG.get("model_profile", "full (default)"), CONFIG.get("models_dir"))
predictor = WellborePredictor(
    models_dir=CONFIG.get("models_dir"),
    profile=CONFIG.get("model_profile"),
    profiles=CONFIG.get("model_profiles"),
    s3_bucket=CONFIG.get("models_s3_bucket"),
)
log.info("Predictor loaded profile '%s': %d LightGBM seeds, %d CatBoost seeds",
          predictor.profile, len(predictor.lgb_models), len(predictor.cat_models))

_default_tw_path = CONFIG.get("type_well_csv")
_default_tw_df = None
if _default_tw_path:
    _default_tw_path = Path(_default_tw_path)
    if _default_tw_path.exists():
        _default_tw_df = pd.read_csv(_default_tw_path)
        log.info("Loaded default type well curve from %s (%d rows)", _default_tw_path, len(_default_tw_df))
    else:
        log.warning("configured type_well_csv %s does not exist", _default_tw_path)


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #

class PredictionPoint(BaseModel):
    id: str
    md: Optional[float] = None
    pred: float


class PredictResponse(BaseModel):
    well: str
    n_points: int
    predictions: List[PredictionPoint]


class BatchError(BaseModel):
    filename: str
    error: str


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]
    errors: List[BatchError]


# --------------------------------------------------------------------------- #
# Core logic (sync, runs in threadpool) -- operates on DataFrames only, no I/O
# --------------------------------------------------------------------------- #

def _well_id_from_filename(filename, override=None):
    if override:
        return override
    stem = Path(filename or "well").stem
    return stem.replace("__horizontal_well", "")


def _run_single_df(hw: pd.DataFrame, tw: pd.DataFrame, wid: str) -> dict:
    feat_df = build_well(hw, tw, predictor.FI, predictor.DI, wid=wid, is_train=False)
    if feat_df is None or feat_df.empty:
        raise ValueError(
            f"Could not build features for well '{wid}' "
            "(not enough known TVT points, no rows to predict, or bad type well curve)."
        )

    output_df = predictor.predict(feat_df)
    if output_df is None or output_df.empty:
        raise ValueError(f"Prediction failed for well '{wid}'.")

    # Recover MD for the predicted rows to make the response more useful.
    ev_idx = hw.index[hw["TVT_input"].isna()]
    mds = hw.loc[ev_idx, "MD"].to_numpy() if "MD" in hw.columns else [None] * len(output_df)

    predictions = [
        {"id": row_id, "md": (float(md) if md is not None else None), "pred": float(pred)}
        for row_id, md, pred in zip(output_df["id"], mds, output_df["pred"])
    ]

    return {"well": wid, "n_points": len(predictions), "predictions": predictions}


async def _read_csv_upload(upload: UploadFile) -> pd.DataFrame:
    raw = await upload.read()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Could not parse '{upload.filename}' as CSV: {e}")


async def _resolve_type_well(type_well_file: Optional[UploadFile]) -> pd.DataFrame:
    if type_well_file is not None:
        return await _read_csv_upload(type_well_file)
    if _default_tw_df is None:
        raise ValueError(
            "No type_well_file uploaded and no default type_well_csv configured in config.yaml"
        )
    return _default_tw_df


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_profile": predictor.profile,
        "lgb_seeds": len(predictor.lgb_models),
        "cat_seeds": len(predictor.cat_models),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Horizontal well CSV to predict on"),
    well_id: Optional[str] = Form(None),
    type_well_file: Optional[UploadFile] = File(None, description="Optional type-well reference CSV"),
):
    try:
        hw = await _read_csv_upload(file)
        tw = await _resolve_type_well(type_well_file)
        wid = _well_id_from_filename(file.filename, well_id)

        return await run_in_threadpool(_run_single_df, hw, tw, wid)
    except ValueError as e:
        log.exception("ValueError in /predict")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail=f"internal error: {e}")


@app.post("/predict_batch", response_model=PredictBatchResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="One or more horizontal well CSVs"),
    type_well_file: Optional[UploadFile] = File(None, description="Optional type-well reference CSV, shared across all wells"),
):
    results, errors = [], []

    try:
        tw = await _resolve_type_well(type_well_file)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    for f in files:
        try:
            hw = await _read_csv_upload(f)
            wid = _well_id_from_filename(f.filename)
            result = await run_in_threadpool(_run_single_df, hw, tw, wid)
            results.append(result)
        except Exception as e:
            errors.append({"filename": f.filename, "error": str(e)})

    return {"results": results, "errors": errors}


if __name__ == "__main__":
    import uvicorn

    reload_enabled = CONFIG.get("reload", False)
    uvicorn.run(
        "app:app" if reload_enabled else app,  # string form (re-import) only needed for --reload
        host=CONFIG.get("host", "0.0.0.0"),
        port=CONFIG.get("port", 8000),
        reload=reload_enabled,
    )
