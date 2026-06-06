from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.config import TOKYO_STATIONS_FILE


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


FALLBACK_STATIONS = [
    {"name": "Tokyo Station", "latitude": 35.681236, "longitude": 139.767125},
    {"name": "Shinjuku Station", "latitude": 35.690921, "longitude": 139.700258},
    {"name": "Shibuya Station", "latitude": 35.658034, "longitude": 139.701636},
    {"name": "Ikebukuro Station", "latitude": 35.729503, "longitude": 139.7109},
    {"name": "Ueno Station", "latitude": 35.713768, "longitude": 139.777254},
    {"name": "Shinagawa Station", "latitude": 35.628471, "longitude": 139.73876},
    {"name": "Akihabara Station", "latitude": 35.698353, "longitude": 139.773114},
    {"name": "Ginza Station", "latitude": 35.671989, "longitude": 139.763965},
    {"name": "Roppongi Station", "latitude": 35.662836, "longitude": 139.731443},
    {"name": "Kichijoji Station", "latitude": 35.703119, "longitude": 139.579813},
]


def fetch_tokyo_stations(output_path: str | Path = TOKYO_STATIONS_FILE) -> pd.DataFrame:
    query = """
    [out:json][timeout:60];
    area["name:en"="Tokyo"]["boundary"="administrative"]->.tokyo;
    (
      node(area.tokyo)["railway"="station"];
      node(area.tokyo)["public_transport"="station"]["station"~"train|subway|light_rail|monorail"];
    );
    out body;
    """
    request = Request(
        OVERPASS_URL,
        data=urlencode({"data": query}).encode("utf-8"),
        headers={"User-Agent": "DataMiningJapanHousePriceDemo/1.0"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = []
        seen = set()
        for item in payload.get("elements", []):
            tags = item.get("tags", {})
            name = tags.get("name:en") or tags.get("name") or tags.get("name:ja")
            lat = item.get("lat")
            lon = item.get("lon")
            if not name or lat is None or lon is None:
                continue
            key = (round(float(lat), 6), round(float(lon), 6), name)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "operator": tags.get("operator", ""),
                    "network": tags.get("network", ""),
                }
            )
        stations = pd.DataFrame(rows)
    except Exception:
        stations = pd.DataFrame(FALLBACK_STATIONS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stations = stations.sort_values("name").drop_duplicates(["name", "latitude", "longitude"])
    stations.to_csv(output_path, index=False)
    return stations


if __name__ == "__main__":
    data = fetch_tokyo_stations()
    print(f"saved {len(data)} stations to {TOKYO_STATIONS_FILE}")
    print(data.head(10).to_string(index=False).encode("ascii", errors="ignore").decode("ascii"))
