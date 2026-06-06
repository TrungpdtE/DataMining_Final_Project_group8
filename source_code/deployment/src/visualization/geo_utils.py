from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def filter_by_year(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    if year is None or "Year" not in df.columns:
        return df.copy()
    return df[df["Year"] == year].copy()


def dataframe_to_geojson(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    max_rows: int | None = 5000,
) -> dict[str, Any]:
    data = df.dropna(subset=[lat_col, lon_col]).copy()
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)

    features = []
    for _, row in data.iterrows():
        properties = row.drop(labels=[lat_col, lon_col]).to_dict()
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row[lon_col]), float(row[lat_col])],
                },
                "properties": _json_safe(properties),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def save_geojson(df: pd.DataFrame, output_path: str | Path, year: int | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = filter_by_year(df, year)
    geojson = dataframe_to_geojson(data)
    output_path.write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")
    return output_path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
