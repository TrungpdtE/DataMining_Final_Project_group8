from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import MODEL_DIR, PREFECTURE_STATIONS_FILE, PROJECT_ROOT, TOKYO_STATIONS_FILE
from src.features.feature_engineering import prepare_model_frame
from src.features.feature_engineering import haversine_km
from src.models.train_xgb import train
from src.models.predict import predict_prices
from src.preprocessing.clean_data import load_clean_merged, load_macro_data
from src.visualization.geo_utils import dataframe_to_geojson, filter_by_year


app = FastAPI(title="Japan House Price Prediction API")
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TRAIN_STATUS: dict[str, Any] = {"state": "idle", "message": "No training job running"}


class MacroInputs(BaseModel):
    gdp_growth: float = 0.0
    interest_rate: float = 0.0
    inflation_rate: float = 0.0
    population_growth: float = 0.0
    housing_price_index: float = 100.0


class PredictRequest(MacroInputs):
    year: int = Field(default=2026, ge=2005, le=2035)
    quarter: int = Field(default=2, ge=1, le=4)
    prefecture: str | None = None
    location: str | None = None
    area: float = Field(default=120, gt=0)
    floor_area: float = Field(default=90, gt=0)
    frontage: float | None = Field(default=None, ge=0)
    coverage: float | None = Field(default=None, ge=0)
    floor_area_ratio: float | None = Field(default=None, ge=0)
    property_type: str | None = None
    construction_year: int = Field(default=2010, ge=1900, le=2035)
    average_time_to_station: float = Field(default=15, ge=0)
    latitude: float | None = Field(default=None, ge=20, le=50)
    longitude: float | None = Field(default=None, ge=120, le=155)


@lru_cache(maxsize=2)
def _load_demo_frame_cached(drop_outliers: bool = False) -> pd.DataFrame:
    _, df = load_clean_merged(drop_outliers=drop_outliers)
    return df


def _load_demo_frame(drop_outliers: bool = False) -> pd.DataFrame:
    return _load_demo_frame_cached(drop_outliers=drop_outliers).copy()


@lru_cache(maxsize=1)
def _load_tokyo_stations() -> pd.DataFrame:
    if PREFECTURE_STATIONS_FILE.exists():
        stations = pd.read_csv(PREFECTURE_STATIONS_FILE)
    elif TOKYO_STATIONS_FILE.exists():
        stations = pd.read_csv(TOKYO_STATIONS_FILE)
        stations["prefecture"] = "Tokyo"
    else:
        from src.preprocessing.fetch_prefecture_stations import fetch_prefecture_stations

        stations = fetch_prefecture_stations()
    if "prefecture" not in stations.columns:
        stations["prefecture"] = "Tokyo"
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    return stations.dropna(subset=["latitude", "longitude"]).copy()


def _apply_macro(df: pd.DataFrame, macro: MacroInputs | None = None) -> pd.DataFrame:
    df = df.copy()
    values = {
        key: getattr(macro, key)
        for key in MacroInputs().model_dump().keys()
    } if macro is not None else MacroInputs().model_dump()
    for col, value in values.items():
        df[col] = value
    return df


def _compact_feature(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties", {})
    keep = {
        "Type",
        "Prefecture",
        "Location",
        "price",
        "predicted_price_yen",
        "Year",
        "Quarter",
        "area",
        "floor_area",
        "frontage",
        "coverage",
        "FloorAreaRatio",
        "house_age",
        "AverageTimeToStation",
        "MunicipalityCategory",
        "density",
        "Migration",
        "Distance_to_designated_city",
        "dist_to_tokyo_km",
        "dist_to_osaka_km",
        "dist_to_nagoya_km",
        "dist_to_fukuoka_km",
        "dist_to_sapporo_km",
        "dist_to_nearest_major_center_km",
        "location_cluster",
        "Prefecture_target_mean",
        "Location_target_mean",
        "Close_to_Tokyo",
        "Close_to_greater_Tokyo_area",
        "Close_to_designated_city_flag",
        "gdp_growth",
        "interest_rate",
        "inflation_rate",
        "population_growth",
        "housing_price_index",
        "last_year_prefecture_avg_price",
        "last_year_prefecture_tx_count",
        "prefecture_price_growth_1y",
        "prefecture_price_growth_3y",
        "last_year_location_avg_price",
        "location_price_growth_1y",
        "last_year_global_avg_price",
        "global_price_growth_1y",
        "global_price_growth_3y",
        "nearest_major_center",
        "Nearest_designated_city",
        "split_role",
    }
    feature["properties"] = {k: v for k, v in props.items() if k in keep}
    return feature


def _filter_valid_japan_coords(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    return data[
        data["latitude"].between(24.0, 46.5)
        & data["longitude"].between(122.0, 153.5)
    ].copy()


def _apply_place_filters(
    df: pd.DataFrame,
    prefecture: str | None = None,
    location: str | None = None,
) -> pd.DataFrame:
    data = df
    if prefecture and prefecture != "ALL":
        data = data[data["Prefecture"] == prefecture]
    if location and location != "ALL":
        data = data[data["Location"] == location]
    return data


def _holdout_year() -> int:
    path = MODEL_DIR / "model_metadata.json"
    if path.exists():
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("holdout_year", 2024))
        except Exception:
            return 2024
    return 2024


def _assign_split_role(df: pd.DataFrame, requested_year: int | None = None) -> pd.DataFrame:
    df = df.copy()
    holdout = _holdout_year()
    if requested_year is not None and requested_year > int(_load_demo_frame()["Year"].max()):
        df["split_role"] = "forecast"
        return df
    df["split_role"] = "train"
    df.loc[df["Year"] == holdout, "split_role"] = "test"
    df.loc[df["Year"] > holdout, "split_role"] = "future_observed"
    return df


def _latest_macro_values(year: int | None = None) -> dict[str, float]:
    macro = load_macro_data()
    if macro.empty:
        return MacroInputs().model_dump()
    if year is not None and (macro["year"] <= year).any():
        row = macro[macro["year"] <= year].sort_values("year").iloc[-1]
    else:
        row = macro.sort_values("year").iloc[-1]
    return {
        "gdp_growth": float(row["gdp_growth"]),
        "interest_rate": float(row["interest_rate"]),
        "inflation_rate": float(row["inflation_rate"]),
        "population_growth": float(row["population_growth"]),
        "housing_price_index": float(row["housing_price_index"]),
    }


def _apply_forecast_lags(df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    history = _load_demo_frame(drop_outliers=False)
    source_years = history.loc[history["Year"] < target_year, "Year"]
    if source_years.empty:
        return df
    source_year = int(source_years.max())
    prev_year = source_year - 1
    three_year = source_year - 3

    result = df.copy()
    global_by_year = history.groupby("Year")["price"].mean()
    global_source = float(global_by_year.get(source_year, global_by_year.mean()))
    global_prev = float(global_by_year.get(prev_year, global_source))
    global_three = float(global_by_year.get(three_year, global_prev))

    pref_source = history[history["Year"] == source_year].groupby("Prefecture")["price"].agg(["mean", "size"])
    pref_prev = history[history["Year"] == prev_year].groupby("Prefecture")["price"].mean()
    pref_three = history[history["Year"] == three_year].groupby("Prefecture")["price"].mean()
    loc_source = history[history["Year"] == source_year].groupby(["Prefecture", "Location"])["price"].mean()
    loc_prev = history[history["Year"] == prev_year].groupby(["Prefecture", "Location"])["price"].mean()

    pref_mean_map = pref_source["mean"].to_dict()
    pref_count_map = pref_source["size"].to_dict()
    result["last_year_prefecture_avg_price"] = result["Prefecture"].map(pref_mean_map).fillna(global_source)
    result["last_year_prefecture_tx_count"] = result["Prefecture"].map(pref_count_map).fillna(0)
    result["last_year_global_avg_price"] = global_source
    result["global_price_growth_1y"] = global_source / global_prev - 1 if global_prev else 0
    result["global_price_growth_3y"] = global_source / global_three - 1 if global_three else 0

    result["prefecture_price_growth_1y"] = result["Prefecture"].map(
        {
            pref: value / pref_prev.get(pref, value) - 1
            for pref, value in pref_mean_map.items()
            if pref_prev.get(pref, value)
        }
    ).fillna(result["global_price_growth_1y"])
    result["prefecture_price_growth_3y"] = result["Prefecture"].map(
        {
            pref: value / pref_three.get(pref, value) - 1
            for pref, value in pref_mean_map.items()
            if pref_three.get(pref, value)
        }
    ).fillna(result["global_price_growth_3y"])

    keys = list(zip(result["Prefecture"], result["Location"]))
    result["last_year_location_avg_price"] = [
        float(loc_source.get(key, pref_mean_map.get(key[0], global_source))) for key in keys
    ]
    result["location_price_growth_1y"] = [
        (
            float(loc_source.get(key)) / float(loc_prev.get(key)) - 1
            if key in loc_source.index and key in loc_prev.index and float(loc_prev.get(key)) != 0
            else float(result["prefecture_price_growth_1y"].iloc[i])
        )
        for i, key in enumerate(keys)
    ]
    return result


def _nearest_candidates(df: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    coords = df.dropna(subset=["latitude", "longitude"]).copy()
    if coords.empty:
        return df
    dlat = coords["latitude"].astype(float) - latitude
    dlon = coords["longitude"].astype(float) - longitude
    nearest_idx = (dlat.pow(2) + dlon.pow(2)).sort_values().head(50).index
    return coords.loc[nearest_idx]


def _nearest_context(latitude: float, longitude: float) -> pd.Series:
    df = _filter_valid_japan_coords(_load_demo_frame(drop_outliers=False))
    candidates = _nearest_candidates(df, latitude, longitude)
    row = candidates.iloc[0].copy()
    row["context_distance_km"] = float(
        haversine_km(
            pd.Series([latitude]),
            pd.Series([longitude]),
            float(row["latitude"]),
            float(row["longitude"]),
        )[0]
    )
    return row


def _nearest_station(latitude: float, longitude: float, prefecture: str | None = None) -> dict[str, Any]:
    stations = _load_tokyo_stations()
    if prefecture and prefecture in set(stations["prefecture"]):
        stations = stations[stations["prefecture"] == prefecture]
    distances = haversine_km(
        stations["latitude"],
        stations["longitude"],
        latitude,
        longitude,
    )
    distance_values = np.asarray(distances, dtype=float)
    idx = int(np.argmin(distance_values))
    row = stations.iloc[idx]
    distance_km = float(distance_values[idx])
    return {
        "name": str(row.get("name", "")),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "distance_km": distance_km,
        "walking_minutes": round(distance_km / 4.8 * 60, 1),
        "prefecture": str(row.get("prefecture", "")),
        "operator": str(row.get("operator", "")),
        "network": str(row.get("network", "")),
    }


def _run_training_job() -> None:
    TRAIN_STATUS.update({"state": "running", "message": "Training models"})
    try:
        metrics = train()
        best = metrics.sort_values("rmse_test_log").iloc[0].to_dict()
        TRAIN_STATUS.update(
            {
                "state": "done",
                "message": "Training finished",
                "best_model": best.get("model"),
                "rmse_test_log": best.get("rmse_test_log"),
                "mae_test_yen": best.get("mae_test_yen"),
            }
        )
    except Exception as exc:
        TRAIN_STATUS.update({"state": "error", "message": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/map")
def map_points(
    year: int | None = Query(default=None, ge=2005, le=2035),
    limit: int = Query(default=5000, ge=0, le=200000),
    prefecture: str | None = None,
    location: str | None = None,
) -> dict:
    df = _load_demo_frame(drop_outliers=False)
    if year is not None and year > int(df["Year"].max()):
        return {"type": "FeatureCollection", "features": []}
    df = prepare_model_frame(df)
    df = filter_by_year(df, year)
    df = _apply_place_filters(df, prefecture, location)
    df = _filter_valid_japan_coords(df)
    df = _assign_split_role(df, year)
    geojson = dataframe_to_geojson(df, max_rows=None if limit == 0 else limit)
    geojson["features"] = [_compact_feature(f) for f in geojson["features"]]
    return geojson


@app.get("/map/predictions")
def map_predictions(
    year: int | None = Query(default=None, ge=2005, le=2035),
    limit: int = Query(default=5000, ge=0, le=200000),
    prefecture: str | None = None,
    location: str | None = None,
    gdp_growth: float | None = None,
    interest_rate: float | None = None,
    inflation_rate: float | None = None,
    population_growth: float | None = None,
    housing_price_index: float | None = None,
) -> dict:
    df = _load_demo_frame(drop_outliers=False)
    max_year = int(df["Year"].max())
    if year is not None and year > max_year:
        df = filter_by_year(df, max_year)
        df = df.copy()
        df["Year"] = year
        df = _apply_forecast_lags(df, year)
    else:
        df = filter_by_year(df, year)
    df = _apply_place_filters(df, prefecture, location)
    df = _filter_valid_japan_coords(df)
    defaults = _latest_macro_values(year)
    df = df.copy()
    for col, value in {
        "gdp_growth": gdp_growth,
        "interest_rate": interest_rate,
        "inflation_rate": inflation_rate,
        "population_growth": population_growth,
        "housing_price_index": housing_price_index,
    }.items():
        df[col] = defaults[col] if value is None else value
    df = predict_prices(df)
    df = _assign_split_role(df, year)
    geojson = dataframe_to_geojson(df, max_rows=None if limit == 0 else limit)
    geojson["features"] = [_compact_feature(f) for f in geojson["features"]]
    return geojson


@app.get("/options")
def options() -> dict[str, Any]:
    df = _load_demo_frame(drop_outliers=False)
    counts = df["Prefecture"].value_counts()
    selected = set(counts.head(12).index)
    locations = (
        df.groupby(["Prefecture", "Location"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    return {
        "years": [int(df["Year"].min()), int(df["Year"].max())],
        "types": [str(value) for value in sorted(df["Type"].dropna().unique())],
        "station_prefectures": sorted(_load_tokyo_stations()["prefecture"].dropna().unique().tolist()),
        "prefectures": [
            {"name": str(name), "count": int(count), "used": name in selected}
            for name, count in counts.items()
        ],
        "locations": locations.to_dict(orient="records"),
    }


@app.get("/macro/latest")
def macro_latest(year: int | None = Query(default=None, ge=2005, le=2035)) -> dict[str, Any]:
    values = _latest_macro_values(year)
    macro = load_macro_data()
    source_year = None
    if not macro.empty:
        source = macro[macro["year"] <= year].sort_values("year") if year else macro.sort_values("year")
        if not source.empty:
            source_year = int(source.iloc[-1]["year"])
    return {"source_year": source_year, **values}


@app.get("/location/context")
def location_context(
    latitude: float = Query(ge=20, le=50),
    longitude: float = Query(ge=120, le=155),
) -> dict[str, Any]:
    row = _nearest_context(latitude, longitude)
    station = _nearest_station(latitude, longitude, str(row.get("Prefecture", "")))
    return {
        "prefecture": str(row.get("Prefecture", "")),
        "location": str(row.get("Location", "")),
        "nearest_dataset_latitude": float(row["latitude"]),
        "nearest_dataset_longitude": float(row["longitude"]),
        "clicked_latitude": latitude,
        "clicked_longitude": longitude,
        "context_distance_km": float(row["context_distance_km"]),
        "nearest_station": station,
        "suggested_average_time_to_station": station["walking_minutes"],
        "suggested_frontage": None if pd.isna(row.get("frontage")) else float(row.get("frontage")),
        "suggested_coverage": None if pd.isna(row.get("coverage")) else float(row.get("coverage")),
        "suggested_floor_area_ratio": None
        if pd.isna(row.get("FloorAreaRatio"))
        else float(row.get("FloorAreaRatio")),
        "suggested_type": str(row.get("Type", "")),
    }


@app.get("/stations")
def stations(
    prefecture: str | None = None,
    limit: int = Query(default=1500, ge=0, le=10000),
) -> dict[str, Any]:
    stations = _load_tokyo_stations()
    if prefecture and prefecture != "ALL":
        stations = stations[stations["prefecture"] == prefecture]
    if limit:
        stations = stations.head(limit)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": {
                    "name": str(row.get("name", "")),
                    "prefecture": str(row.get("prefecture", "")),
                    "operator": str(row.get("operator", "")),
                    "network": str(row.get("network", "")),
                },
            }
            for _, row in stations.iterrows()
        ],
    }


@app.get("/stations/tokyo")
def tokyo_stations(limit: int = Query(default=1000, ge=10, le=2000)) -> dict[str, Any]:
    return stations(prefecture="Tokyo", limit=limit)


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    df = _load_demo_frame(drop_outliers=False)
    candidates = df
    if payload.prefecture:
        candidates = candidates[candidates["Prefecture"] == payload.prefecture]
    if payload.location:
        narrowed = candidates[candidates["Location"] == payload.location]
        if not narrowed.empty:
            candidates = narrowed
    if candidates.empty:
        candidates = df
    if payload.latitude is not None and payload.longitude is not None:
        candidates = _nearest_candidates(candidates, payload.latitude, payload.longitude)
    with_coords = candidates.dropna(subset=["latitude", "longitude"])
    if not with_coords.empty:
        candidates = with_coords

    row = candidates.sample(1, random_state=42).copy()
    row["Year"] = payload.year
    row["Quarter"] = payload.quarter
    row["area"] = payload.area
    row["floor_area"] = payload.floor_area
    row["density"] = payload.floor_area / payload.area if payload.area else 0
    if payload.frontage is not None:
        row["frontage"] = payload.frontage
    if payload.coverage is not None:
        row["coverage"] = payload.coverage
    if payload.floor_area_ratio is not None:
        row["FloorAreaRatio"] = payload.floor_area_ratio
    if payload.property_type:
        row["Type"] = payload.property_type
    row["construction_year"] = payload.construction_year
    row["AverageTimeToStation"] = payload.average_time_to_station
    if payload.latitude is not None and payload.longitude is not None:
        row["latitude"] = payload.latitude
        row["longitude"] = payload.longitude
    if payload.year > int(df["Year"].max()):
        row = _apply_forecast_lags(row, payload.year)
    row = _apply_macro(row, payload)
    predicted = predict_prices(row).iloc[0]
    return {
        "estimated_price_yen": float(predicted["predicted_price_yen"]),
        "estimated_price_million_yen": float(predicted["predicted_price_yen"] / 1_000_000),
        "year": payload.year,
        "prefecture": str(predicted.get("Prefecture", "")),
        "location": str(predicted.get("Location", "")),
        "latitude": None if pd.isna(predicted.get("latitude")) else float(predicted.get("latitude")),
        "longitude": None if pd.isna(predicted.get("longitude")) else float(predicted.get("longitude")),
    }


@app.post("/predict/timeline")
def predict_timeline(payload: PredictRequest) -> dict[str, Any]:
    start_year = max(2020, payload.year - 3)
    end_year = min(2035, payload.year + 3)
    points = []
    for forecast_year in range(start_year, end_year + 1):
        year_payload = payload.model_copy(update={"year": forecast_year})
        result = predict(year_payload)
        macro = _latest_macro_values(forecast_year)
        points.append(
            {
                "year": forecast_year,
                "estimated_price_yen": result["estimated_price_yen"],
                "estimated_price_million_yen": result["estimated_price_million_yen"],
                "macro": macro,
            }
        )
    return {"points": points}


@app.post("/train")
def train_models(background_tasks: BackgroundTasks) -> dict[str, str]:
    if TRAIN_STATUS.get("state") == "running":
        return {"state": "running", "message": "Training is already running"}
    background_tasks.add_task(_run_training_job)
    TRAIN_STATUS.update({"state": "queued", "message": "Training queued"})
    return {"state": "queued", "message": "Training queued"}


@app.get("/train/status")
def train_status() -> dict[str, Any]:
    return TRAIN_STATUS


@app.get("/model/metadata")
def model_metadata() -> dict[str, Any]:
    path = MODEL_DIR / "model_metadata.json"
    if not path.exists():
        return {"message": "No saved metadata yet"}
    return json.loads(path.read_text(encoding="utf-8"))


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)
