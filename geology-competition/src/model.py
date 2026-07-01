import os
import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pathlib import Path

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

    def predict(self, feat_df):

        if feat_df is None or feat_df.empty:
            return None
            
        features_to_use = [c for c in feat_df.columns if c not in {'well', 'id', 'target'}]
        X = feat_df[features_to_use]
  
        all_seed_preds = []

        for folds in self.lgb_models:
            lgb_seed_pred = np.zeros(len(X), dtype=np.float32)
            for m in folds:
                best_iter = getattr(m, 'best_iteration', None)
                lgb_seed_pred += m.predict(X, num_iteration=best_iter).astype(np.float32) / len(folds)
            all_seed_preds.append(lgb_seed_pred)

        for folds in self.cat_models:
            cb_seed_pred = np.zeros(len(X), dtype=np.float32)
            for m in folds:
                cb_seed_pred += m.predict(X.values).astype(np.float32) / len(folds)
            all_seed_preds.append(cb_seed_pred)

        final_offset_pred = np.stack(all_seed_preds, axis=1).mean(axis=1)

        output_df = feat_df.copy()
        output_df['pred'] = output_df['last_known_tvt'].values + final_offset_pred

        output_df = self.sg_smooth(output_df, col_name='pred')

        return output_df