import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pathlib import Path

from src.features import FormationPlaneKNN, DenseANCCImputer
import __main__
__main__.FormationPlaneKNN = FormationPlaneKNN
__main__.DenseANCCImputer = DenseANCCImputer


class WellborePredictor:
    def __init__(self, models_dir= None):

        if models_dir is None:
            current_dir = Path(__file__).resolve().parent
            self.models_dir = current_dir / "models"
        else:
            self.models_dir = Path(models_dir)

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