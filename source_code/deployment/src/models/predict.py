from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR
from src.features.feature_engineering import prepare_model_frame


def load_bundle(model_path: str | Path = MODEL_DIR / "xgb_pipeline.pkl") -> dict:
    return joblib.load(model_path)


def predict_prices(df: pd.DataFrame, model_path: str | Path = MODEL_DIR / "xgb_pipeline.pkl") -> pd.DataFrame:
    bundle = load_bundle(model_path)
    data = prepare_model_frame(df)
    data = bundle["clusterer"].transform(data)
    data = bundle["target_encoder"].transform(data)
    feature_columns = bundle["feature_columns"]
    pred_log = bundle["model"].predict(data[feature_columns])
    result = df.copy()
    result["predicted_log_price"] = pred_log
    result["predicted_price_yen"] = np.expm1(pred_log)
    return result
