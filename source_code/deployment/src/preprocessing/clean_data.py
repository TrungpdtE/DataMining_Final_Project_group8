from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BUILDINGS_FILE, FRED_FILE, LOCATION_FILE, MACRO_FILE


COLUMN_RENAME = {
    "TotalTransactionValue": "price",
    "Area": "area",
    "TotalFloorArea": "floor_area",
    "Frontage": "frontage",
    "ConstructionYear": "construction_year",
    "BuildingCoverageRatio": "coverage",
}

NUMERIC_COLUMNS = [
    "price",
    "area",
    "floor_area",
    "frontage",
    "construction_year",
    "coverage",
    "FloorAreaRatio",
    "Quarter",
    "Year",
    "AverageTimeToStation",
    "MunicipalityCategory",
    "Migration",
]


def _normalize_municipality_code(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64").astype("string")


def load_buildings(path: str | Path = BUILDINGS_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=COLUMN_RENAME)
    if "City,Town,Ward,Village code" in df.columns:
        df["municipality_code"] = _normalize_municipality_code(
            df["City,Town,Ward,Village code"]
        )
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_location_data(path: str | Path = LOCATION_FILE) -> pd.DataFrame:
    location = pd.read_csv(path, low_memory=False)
    location["municipality_code"] = _normalize_municipality_code(
        location["City,Town,Ward,Village code"]
    )
    for col in [
        "latitude",
        "longitude",
        "Distance_to_designated_city",
        "Close_to_Tokyo",
        "Close_to_greater_Tokyo_area",
        "Close_to_designated_city_flag",
    ]:
        if col in location.columns:
            if location[col].dtype == "object" and location[col].isin(["True", "False"]).any():
                location[col] = location[col].map({"True": True, "False": False})
            else:
                converted = pd.to_numeric(location[col], errors="coerce")
                location[col] = converted if converted.notna().any() else location[col]
    keep = [
        "municipality_code",
        "latitude",
        "longitude",
        "Distance_to_designated_city",
        "Nearest_designated_city",
        "Close_to_Tokyo",
        "Close_to_greater_Tokyo_area",
        "Close_to_designated_city_flag",
    ]
    return location[[c for c in keep if c in location.columns]].drop_duplicates(
        "municipality_code"
    )


def load_macro_data(
    macro_path: str | Path = MACRO_FILE,
    fred_path: str | Path = FRED_FILE,
) -> pd.DataFrame:
    macro_path = Path(macro_path)
    if macro_path.exists():
        macro = pd.read_csv(macro_path)
        macro = macro.rename(columns={"Year": "year"})
    else:
        macro = pd.DataFrame(columns=["year"])

    if Path(fred_path).exists():
        fred = pd.read_csv(fred_path)
        fred["observation_date"] = pd.to_datetime(
            fred["observation_date"], errors="coerce"
        )
        fred["year"] = fred["observation_date"].dt.year
        fred["housing_price_index"] = pd.to_numeric(
            fred["QJPN628BIS"], errors="coerce"
        )
        housing = (
            fred.dropna(subset=["year", "housing_price_index"])
            .groupby("year", as_index=False)["housing_price_index"]
            .mean()
        )
        if "housing_price_index" in macro.columns:
            macro = macro.drop(columns=["housing_price_index"])
        macro = macro.merge(housing, on="year", how="outer")

    defaults = {
        "gdp_growth": 0.0,
        "interest_rate": 0.0,
        "inflation_rate": 0.0,
        "population_growth": 0.0,
        "housing_price_index": 100.0,
    }
    if "year" not in macro.columns:
        macro["year"] = []
    macro["year"] = pd.to_numeric(macro["year"], errors="coerce")
    macro = macro.dropna(subset=["year"]).copy()
    macro["year"] = macro["year"].astype(int)
    for col, default in defaults.items():
        if col not in macro.columns:
            macro[col] = default
        macro[col] = pd.to_numeric(macro[col], errors="coerce")
    macro = macro.sort_values("year")
    macro[list(defaults)] = macro[list(defaults)].ffill().bfill().fillna(defaults)
    return macro[["year", *defaults.keys()]].drop_duplicates("year")


def clean_base_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df[df["price"].notna() & (df["price"] > 0)]

    numeric_fill = [
        "area",
        "floor_area",
        "frontage",
        "construction_year",
        "coverage",
        "AverageTimeToStation",
        "Migration",
        "MunicipalityCategory",
    ]
    for col in numeric_fill:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in ["Prefecture", "Location", "Type", "Nearest_designated_city"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    if "construction_year" in df.columns:
        df["house_age"] = (df["Year"] - df["construction_year"]).clip(0, 100)

    df["density"] = df["floor_area"] / df["area"].replace(0, np.nan)
    df["density"] = df["density"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def merge_location_features(
    buildings: pd.DataFrame, location: pd.DataFrame
) -> pd.DataFrame:
    df = buildings.merge(location, on="municipality_code", how="left")
    return df


def merge_macro_features(buildings: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    if macro.empty:
        return buildings
    df = buildings.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    merged = df.merge(macro, left_on="Year", right_on="year", how="left")
    return merged.drop(columns=["year"], errors="ignore")


def add_temporal_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    global_by_year = df.groupby("Year")["price"].mean().sort_index()
    global_prev = global_by_year.shift(1)
    global_growth_1y = global_by_year.pct_change(1).shift(1)
    global_growth_3y = global_by_year.pct_change(3).shift(1)

    prefecture_year = (
        df.groupby(["Prefecture", "Year"])["price"]
        .agg(prefecture_year_avg_price="mean", prefecture_year_tx_count="size")
        .reset_index()
        .sort_values(["Prefecture", "Year"])
    )
    prefecture_year["last_year_prefecture_avg_price"] = prefecture_year.groupby(
        "Prefecture"
    )["prefecture_year_avg_price"].shift(1)
    prefecture_year["last_year_prefecture_tx_count"] = prefecture_year.groupby(
        "Prefecture"
    )["prefecture_year_tx_count"].shift(1)
    prefecture_year["prefecture_price_growth_1y"] = prefecture_year.groupby(
        "Prefecture"
    )["prefecture_year_avg_price"].pct_change(1).shift(1)
    prefecture_year["prefecture_price_growth_3y"] = prefecture_year.groupby(
        "Prefecture"
    )["prefecture_year_avg_price"].pct_change(3).shift(1)
    prefecture_keep = [
        "Prefecture",
        "Year",
        "last_year_prefecture_avg_price",
        "last_year_prefecture_tx_count",
        "prefecture_price_growth_1y",
        "prefecture_price_growth_3y",
    ]
    df = df.merge(prefecture_year[prefecture_keep], on=["Prefecture", "Year"], how="left")

    location_year = (
        df.groupby(["Prefecture", "Location", "Year"])["price"]
        .mean()
        .reset_index(name="location_year_avg_price")
        .sort_values(["Prefecture", "Location", "Year"])
    )
    location_year["last_year_location_avg_price"] = location_year.groupby(
        ["Prefecture", "Location"]
    )["location_year_avg_price"].shift(1)
    location_year["location_price_growth_1y"] = location_year.groupby(
        ["Prefecture", "Location"]
    )["location_year_avg_price"].pct_change(1).shift(1)
    location_keep = [
        "Prefecture",
        "Location",
        "Year",
        "last_year_location_avg_price",
        "location_price_growth_1y",
    ]
    df = df.merge(location_year[location_keep], on=["Prefecture", "Location", "Year"], how="left")

    df["last_year_global_avg_price"] = df["Year"].map(global_prev)
    df["global_price_growth_1y"] = df["Year"].map(global_growth_1y)
    df["global_price_growth_3y"] = df["Year"].map(global_growth_3y)

    lag_price_cols = [
        "last_year_prefecture_avg_price",
        "last_year_location_avg_price",
        "last_year_global_avg_price",
    ]
    for col in lag_price_cols:
        df[col] = df[col].fillna(df["last_year_global_avg_price"])
        df[col] = df[col].fillna(global_by_year.mean())

    growth_cols = [
        "prefecture_price_growth_1y",
        "prefecture_price_growth_3y",
        "location_price_growth_1y",
        "global_price_growth_1y",
        "global_price_growth_3y",
    ]
    for col in growth_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["last_year_prefecture_tx_count"] = df["last_year_prefecture_tx_count"].fillna(0)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rules = {
        "area": df["area"] < 2000,
        "price": df["price"] < df["price"].quantile(0.99),
        "Migration": df["Migration"] < df["Migration"].quantile(0.99),
        "density": df["density"] < df["density"].quantile(0.99),
    }
    mask = np.logical_and.reduce([rule.to_numpy() for rule in rules.values()])
    return df.loc[mask].copy()


def load_clean_merged(
    buildings_path: str | Path = BUILDINGS_FILE,
    location_path: str | Path = LOCATION_FILE,
    macro_path: str | Path = MACRO_FILE,
    drop_outliers: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    buildings = load_buildings(buildings_path)
    location = load_location_data(location_path)
    macro = load_macro_data(macro_path=macro_path)
    cleaned = clean_base_data(buildings)
    merged = merge_location_features(cleaned, location)
    merged = merge_macro_features(merged, macro)
    merged = add_temporal_lag_features(merged)
    if drop_outliers:
        merged = remove_outliers(merged)
    return cleaned, merged


if __name__ == "__main__":
    before, after = load_clean_merged()
    print(f"Before outlier removal: {before.shape}")
    print(f"After merge/location/outlier removal: {after.shape}")
