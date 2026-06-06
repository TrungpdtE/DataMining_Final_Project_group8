# Frontend map

Static React + Leaflet demo served by FastAPI.

Run from project root:

```powershell
$env:PYTHONPATH="."
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

The page calls:

- `GET /options`
- `GET /map`
- `GET /map/predictions`
- `POST /predict`
- `POST /train`
- `GET /macro/latest`

Features:

- Points/heatmap/both layers.
- Click map to predict at a coordinate.
- Timeline historical/predicted view.
