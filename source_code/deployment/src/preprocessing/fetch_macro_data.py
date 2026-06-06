from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from src.config import FRED_FILE, MACRO_FILE


WORLD_BANK_INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation_rate": "FP.CPI.TOTL.ZG",
    "population_growth": "SP.POP.GROW",
    "interest_rate": "FR.INR.RINR",
}


def fetch_world_bank_indicator(indicator: str, start_year: int = 2005) -> pd.DataFrame:
    params = urlencode({"format": "json", "per_page": 200, "date": f"{start_year}:2035"})
    url = f"https://api.worldbank.org/v2/country/JPN/indicator/{indicator}?{params}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload[1] if len(payload) > 1 and payload[1] else []
    data = [
        {"year": int(row["date"]), "value": row["value"]}
        for row in rows
        if row.get("date") and row.get("value") is not None
    ]
    return pd.DataFrame(data)


def load_annual_house_price_index(path: str | Path = FRED_FILE) -> pd.DataFrame:
    fred = pd.read_csv(path)
    fred["observation_date"] = pd.to_datetime(fred["observation_date"], errors="coerce")
    fred["year"] = fred["observation_date"].dt.year
    fred["housing_price_index"] = pd.to_numeric(fred["QJPN628BIS"], errors="coerce")
    annual = (
        fred.dropna(subset=["year", "housing_price_index"])
        .groupby("year", as_index=False)["housing_price_index"]
        .mean()
    )
    annual["year"] = annual["year"].astype(int)
    return annual


def build_macro_dataset(output_path: str | Path = MACRO_FILE) -> pd.DataFrame:
    macro = None
    for column, indicator in WORLD_BANK_INDICATORS.items():
        frame = fetch_world_bank_indicator(indicator).rename(columns={"value": column})
        macro = frame if macro is None else macro.merge(frame, on="year", how="outer")

    housing = load_annual_house_price_index()
    macro = macro.merge(housing, on="year", how="outer")
    macro = macro.sort_values("year")
    numeric_cols = [col for col in macro.columns if col != "year"]
    macro[numeric_cols] = macro[numeric_cols].apply(pd.to_numeric, errors="coerce")
    macro[numeric_cols] = macro[numeric_cols].ffill().bfill()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro.to_csv(output_path, index=False)
    return macro


if __name__ == "__main__":
    data = build_macro_dataset()
    print(data.tail(8).to_string(index=False))
