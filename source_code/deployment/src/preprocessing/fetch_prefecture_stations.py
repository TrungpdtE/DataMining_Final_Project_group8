from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.config import PREFECTURE_STATIONS_FILE


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

TARGET_PREFECTURES = [
    "Tokyo",
    "Kanagawa Prefecture",
    "Osaka Prefecture",
    "Saitama Prefecture",
    "Chiba Prefecture",
    "Aichi Prefecture",
    "Hyogo Prefecture",
    "Hokkaido",
    "Fukuoka Prefecture",
    "Kyoto Prefecture",
]

ALL_JAPAN_PREFECTURES = [
    "Hokkaido",
    "Aomori Prefecture",
    "Iwate Prefecture",
    "Miyagi Prefecture",
    "Akita Prefecture",
    "Yamagata Prefecture",
    "Fukushima Prefecture",
    "Ibaraki Prefecture",
    "Tochigi Prefecture",
    "Gunma Prefecture",
    "Saitama Prefecture",
    "Chiba Prefecture",
    "Tokyo",
    "Kanagawa Prefecture",
    "Niigata Prefecture",
    "Toyama Prefecture",
    "Ishikawa Prefecture",
    "Fukui Prefecture",
    "Yamanashi Prefecture",
    "Nagano Prefecture",
    "Gifu Prefecture",
    "Shizuoka Prefecture",
    "Aichi Prefecture",
    "Mie Prefecture",
    "Shiga Prefecture",
    "Kyoto Prefecture",
    "Osaka Prefecture",
    "Hyogo Prefecture",
    "Nara Prefecture",
    "Wakayama Prefecture",
    "Tottori Prefecture",
    "Shimane Prefecture",
    "Okayama Prefecture",
    "Hiroshima Prefecture",
    "Yamaguchi Prefecture",
    "Tokushima Prefecture",
    "Kagawa Prefecture",
    "Ehime Prefecture",
    "Kochi Prefecture",
    "Fukuoka Prefecture",
    "Saga Prefecture",
    "Nagasaki Prefecture",
    "Kumamoto Prefecture",
    "Oita Prefecture",
    "Miyazaki Prefecture",
    "Kagoshima Prefecture",
    "Okinawa Prefecture",
]

OSM_AREA_NAMES = {
    "Tokyo": ["Tokyo", "Tokyo Metropolis", "東京都"],
    "Kanagawa Prefecture": ["Kanagawa Prefecture", "神奈川県"],
    "Osaka Prefecture": ["Osaka Prefecture", "大阪府"],
    "Saitama Prefecture": ["Saitama Prefecture", "埼玉県"],
    "Chiba Prefecture": ["Chiba Prefecture", "千葉県"],
    "Aichi Prefecture": ["Aichi Prefecture", "愛知県"],
    "Hyogo Prefecture": ["Hyogo Prefecture", "兵庫県"],
    "Hokkaido": ["Hokkaido", "北海道"],
    "Fukuoka Prefecture": ["Fukuoka Prefecture", "福岡県"],
    "Kyoto Prefecture": ["Kyoto Prefecture", "京都府"],
}


FALLBACK_STATIONS = [
    {"prefecture": "Tokyo", "name": "Tokyo Station", "latitude": 35.681236, "longitude": 139.767125},
    {"prefecture": "Tokyo", "name": "Shinjuku Station", "latitude": 35.690921, "longitude": 139.700258},
    {"prefecture": "Kanagawa Prefecture", "name": "Yokohama Station", "latitude": 35.465833, "longitude": 139.622778},
    {"prefecture": "Osaka Prefecture", "name": "Osaka Station", "latitude": 34.702485, "longitude": 135.495951},
    {"prefecture": "Saitama Prefecture", "name": "Omiya Station", "latitude": 35.906389, "longitude": 139.623889},
    {"prefecture": "Chiba Prefecture", "name": "Chiba Station", "latitude": 35.613056, "longitude": 140.112778},
    {"prefecture": "Aichi Prefecture", "name": "Nagoya Station", "latitude": 35.170915, "longitude": 136.881537},
    {"prefecture": "Hyogo Prefecture", "name": "Sannomiya Station", "latitude": 34.694722, "longitude": 135.195278},
    {"prefecture": "Hokkaido", "name": "Sapporo Station", "latitude": 43.068661, "longitude": 141.350755},
    {"prefecture": "Fukuoka Prefecture", "name": "Hakata Station", "latitude": 33.590355, "longitude": 130.420609},
    {"prefecture": "Kyoto Prefecture", "name": "Kyoto Station", "latitude": 34.985849, "longitude": 135.758766},
]


def _area_selector(prefecture: str) -> str:
    names = OSM_AREA_NAMES.get(prefecture, [prefecture])
    selectors = []
    for name in names:
        selectors.append(f'area["name:en"="{name}"]["boundary"="administrative"]')
        selectors.append(f'area["name"="{name}"]["boundary"="administrative"]')
    return ";".join(selectors)


def _fetch_one(prefecture: str) -> pd.DataFrame:
    query = f"""
    [out:json][timeout:90];
    (
      {_area_selector(prefecture)};
    )->.pref;
    (
      nwr(area.pref)["railway"="station"];
      nwr(area.pref)["public_transport"="station"]["station"~"train|subway|light_rail|monorail"];
    );
    out center tags;
    """
    request = Request(
        OVERPASS_URL,
        data=urlencode({"data": query}).encode("utf-8"),
        headers={"User-Agent": "DataMiningJapanHousePriceDemo/1.0"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    seen = set()
    for item in payload.get("elements", []):
        tags = item.get("tags", {})
        name = tags.get("name:en") or tags.get("name") or tags.get("name:ja")
        center = item.get("center") or {}
        lat = item.get("lat", center.get("lat"))
        lon = item.get("lon", center.get("lon"))
        if not name or lat is None or lon is None:
            continue
        key = (prefecture, round(float(lat), 6), round(float(lon), 6), name)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "prefecture": prefecture,
                "name": name,
                "latitude": float(lat),
                "longitude": float(lon),
                "operator": tags.get("operator", ""),
                "network": tags.get("network", ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_prefecture_stations(
    prefectures: list[str] | None = None,
    output_path: str | Path = PREFECTURE_STATIONS_FILE,
) -> pd.DataFrame:
    prefectures = prefectures or ALL_JAPAN_PREFECTURES
    frames = []
    for prefecture in prefectures:
        try:
            frame = _fetch_one(prefecture)
            if frame.empty:
                raise RuntimeError(f"No stations returned for {prefecture}")
            frames.append(frame)
            print(f"{prefecture}: {len(frame)} stations")
            time.sleep(1.0)
        except Exception as exc:
            fallback = pd.DataFrame([row for row in FALLBACK_STATIONS if row["prefecture"] == prefecture])
            frames.append(fallback)
            print(f"{prefecture}: fallback used ({exc})")

    stations = pd.concat(frames, ignore_index=True)
    stations = stations.sort_values(["prefecture", "name"]).drop_duplicates(
        ["prefecture", "name", "latitude", "longitude"]
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(output_path, index=False)
    return stations


if __name__ == "__main__":
    data = fetch_prefecture_stations()
    print(f"saved {len(data)} stations to {PREFECTURE_STATIONS_FILE}")
