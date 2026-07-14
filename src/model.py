import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.features import FormationPlaneKNN, DenseANCCImputer
import __main__
__main__.FormationPlaneKNN = FormationPlaneKNN
__main__.DenseANCCImputer = DenseANCCImputer

log = logging.getLogger(__name__)

# The exact set of model artifacts WellborePredictor needs to run.
REQUIRED_MODEL_FILES = [
    "lightgbm-1.pkl", "lightgbm-2.pkl", "lightgbm-3.pkl",
    "catboost-1.pkl", "catboost-2.pkl", "catboost-3.pkl",
    "FI_knn.pkl", "DI_imputer.pkl",
]

# Defaults match the standalone download_models.py script; override via env vars
# (MODELS_S3_BUCKET / MODELS_S3_KEY) or the WellborePredictor(...) constructor args
# without having to touch code.
DEFAULT_S3_BUCKET = "wellbore-geology-models"
DEFAULT_S3_KEY = "models.zip"


def _download_models_from_s3(models_dir: Path, bucket: str, key: str):
    """Download models.zip from S3 and populate models_dir with its contents.

    Extracts to a temp dir first (rather than models_dir directly) so this works
    regardless of whether the zip stores the .pkl files at its root or nested
    inside a folder -- we just locate every .pkl after extraction and copy it
    into models_dir, flattening any nesting.
    """
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError(
            "boto3 is required to auto-download models from S3 but isn't installed. "
            "Run `pip install boto3` (and add it to requirements.txt)."
        ) from e

    models_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading models from s3://%s/%s ...", bucket, key)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "models.zip"
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(zip_path))

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)

        extracted_pkls = list(Path(tmp).rglob("*.pkl"))
        if not extracted_pkls:
            raise RuntimeError(f"No .pkl files found inside s3://{bucket}/{key} after extraction.")

        for f in extracted_pkls:
            shutil.copy2(f, models_dir / f.name)

    log.info("Downloaded and extracted %d model file(s) into %s", len(extracted_pkls), models_dir)


class WellborePredictor:
    def __init__(self, models_dir=None, s3_bucket=None, s3_key=None, auto_download=True):

        if models_dir is None:
            current_dir = Path(__file__).resolve().parent
            self.models_dir = current_dir / "models"
        else:
            self.models_dir = Path(models_dir)

        missing = [f for f in REQUIRED_MODEL_FILES if not (self.models_dir / f).exists()]
        if missing:
            if not auto_download:
                raise FileNotFoundError(
                    f"Missing model file(s) {missing} in {self.models_dir} and auto_download=False."
                )
            bucket = s3_bucket or os.environ.get("MODELS_S3_BUCKET", DEFAULT_S3_BUCKET)
            key = s3_key or os.environ.get("MODELS_S3_KEY", DEFAULT_S3_KEY)
            log.warning("Missing model file(s) %s in %s -- attempting S3 download.", missing, self.models_dir)
            _download_models_from_s3(self.models_dir, bucket, key)

            still_missing = [f for f in REQUIRED_MODEL_FILES if not (self.models_dir / f).exists()]
            if still_missing:
                raise FileNotFoundError(
                    f"Still missing model file(s) {still_missing} in {self.models_dir} after S3 download."
                )

        self.lgb_models = [joblib.load(self.models_dir / f"lightgbm-{i}.pkl") for i in range(1, 4)]
        self.cat_models = [joblib.load(self.models_dir / f"catboost-{i}.pkl") for i in range(1, 4)]
        self.FI = joblib.load(self.models_dir / "FI_knn.pkl")
        self.DI = joblib.load(self.models_dir / "DI_imputer.pkl")

    def sg_smooth(self, df, col_name='pred', sg_w=17, sg_p=3):
  
        df = df.copy()
        for _, g in df.groupby('well', sort=False):
            v = g[col_name].values
            n = len(v)
            wl = min(sg_w, n)
            if wl % 2 == 0: 
                wl -= 1
            if wl >= sg_p + 2: 
                v = savgol_filter(v, wl, sg_p)
            df.loc[g.index, col_name] = v
        return df

    def _select_features(self, feat_df, feature_names, model_label):
        missing = [c for c in feature_names if c not in feat_df.columns]
        if missing:
            raise ValueError(
                f"feat_df is missing {len(missing)} feature(s) required by {model_label}: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        return feat_df[feature_names]

    def _prepare_model_inputs(self, feat_df):
        """Build the exact feature matrix each individual model needs (name + order),
        validated up front, before any inference runs.

        CatBoost's models here only expose generic feature names ('0', '1', ...)
        because they were trained on a plain numpy array (no column names attached).
        Since this stacking pipeline trains LightGBM and CatBoost on the same
        underlying feature matrix, we reuse LightGBM's real feature name/order as
        the reference for CatBoost too, instead of CatBoost's own placeholder names.

        NOTE: this assumes LightGBM and CatBoost were trained on identical columns
        in identical order. If that's not actually true for this pipeline, CatBoost
        would silently receive the wrong columns with no error raised (since the
        names would still "exist" in feat_df) -- worth double-checking against the
        training notebook if predictions look off.
        """
        reference_features = self.lgb_models[0][0].feature_name()

        lgb_inputs = [
            [self._select_features(feat_df, m.feature_name(), f"LightGBM seed {i}") for m in folds]
            for i, folds in enumerate(self.lgb_models, start=1)
        ]
        cat_inputs = [
            [self._select_features(feat_df, reference_features, f"CatBoost seed {i}") for m in folds]
            for i, folds in enumerate(self.cat_models, start=1)
        ]
        return lgb_inputs, cat_inputs

    def predict(self, feat_df):

        if feat_df is None or feat_df.empty:
            return None

        lgb_inputs, cat_inputs = self._prepare_model_inputs(feat_df)

        all_seed_preds = []

        for folds, X_folds in zip(self.lgb_models, lgb_inputs):
            lgb_seed_pred = np.zeros(len(feat_df), dtype=np.float32)
            for m, X in zip(folds, X_folds):
                best_iter = getattr(m, 'best_iteration', None)
                lgb_seed_pred += m.predict(X, num_iteration=best_iter).astype(np.float32) / len(folds)
            all_seed_preds.append(lgb_seed_pred)

        for folds, X_folds in zip(self.cat_models, cat_inputs):
            cb_seed_pred = np.zeros(len(feat_df), dtype=np.float32)
            for m, X in zip(folds, X_folds):
                cb_seed_pred += m.predict(X.values).astype(np.float32) / len(folds)
            all_seed_preds.append(cb_seed_pred)

        final_offset_pred = np.stack(all_seed_preds, axis=1).mean(axis=1)

        output_df = feat_df.copy()
        output_df['pred'] = output_df['last_known_tvt'].values + final_offset_pred

        output_df = self.sg_smooth(output_df, col_name='pred')

        return output_df
