from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from src.config import RANDOM_STATE


MAJOR_CENTERS = {
    "tokyo": (35.681236, 139.767125),
    "osaka": (34.702485, 135.495951),
    "nagoya": (35.170915, 136.881537),
    "fukuoka": (33.590355, 130.401716),
    "sapporo": (43.068661, 141.350755),
}


def haversine_km(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: float,
    lon2: float,
) -> np.ndarray:
    radius = 6371.0088
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(a))


def add_location_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    lat_median = df["latitude"].median()
    lon_median = df["longitude"].median()
    if pd.isna(lat_median):
        lat_median = 36.2048
    if pd.isna(lon_median):
        lon_median = 138.2529
    df["latitude"] = df["latitude"].fillna(lat_median)
    df["longitude"] = df["longitude"].fillna(lon_median)

    distance_cols = []
    for name, (lat, lon) in MAJOR_CENTERS.items():
        col = f"dist_to_{name}_km"
        df[col] = haversine_km(df["latitude"], df["longitude"], lat, lon)
        distance_cols.append(col)

    df["dist_to_nearest_major_center_km"] = df[distance_cols].min(axis=1)
    df["nearest_major_center"] = (
        df[distance_cols].idxmin(axis=1).str.replace("dist_to_", "", regex=False).str.replace("_km", "", regex=False)
    )
    df["log_dist_to_tokyo"] = np.log1p(df["dist_to_tokyo_km"])
    df["log_dist_to_nearest_major_center"] = np.log1p(
        df["dist_to_nearest_major_center_km"]
    )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Quarter" in df.columns:
        df["quarter_sin"] = np.sin(2 * np.pi * df["Quarter"] / 4)
        df["quarter_cos"] = np.cos(2 * np.pi * df["Quarter"] / 4)
    return df


class LocationClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 12, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        coords = X[["latitude", "longitude"]].copy()
        self.fill_values_ = coords.median()
        coords = coords.fillna(self.fill_values_)
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans_.fit(coords)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        coords = X[["latitude", "longitude"]].fillna(self.fill_values_)
        X["location_cluster"] = self.kmeans_.predict(coords)
        return X


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], smoothing: float = 20.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = float(y.mean())
        self.maps_ = {}
        data = X[self.columns].copy()
        data["_target"] = y.to_numpy()
        for col in self.columns:
            stats = data.groupby(col)["_target"].agg(["mean", "count"])
            weight = stats["count"] / (stats["count"] + self.smoothing)
            encoded = self.global_mean_ * (1 - weight) + stats["mean"] * weight
            self.maps_[col] = encoded.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            X[f"{col}_target_mean"] = X[col].map(self.maps_[col]).fillna(
                self.global_mean_
            )
        return X


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_location_distance_features(df)
    df = add_time_features(df)
    if "Year" in df.columns and "construction_year" in df.columns:
        df["house_age"] = (
            pd.to_numeric(df["Year"], errors="coerce")
            - pd.to_numeric(df["construction_year"], errors="coerce")
        ).clip(0, 100)
    for col in [
        "Close_to_Tokyo",
        "Close_to_greater_Tokyo_area",
        "Close_to_designated_city_flag",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("boolean").fillna(False).astype(int)
    return df
