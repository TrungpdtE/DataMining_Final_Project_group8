from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "dataset_jp_house"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "saved_models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

BUILDINGS_FILE = RAW_DATA_DIR / "All_prefectures_buildings_with_migration.csv"
LOCATION_FILE = RAW_DATA_DIR / "Location_Data.csv"
FRED_FILE = RAW_DATA_DIR / "QJPN628BIS.csv"
MACRO_FILE = RAW_DATA_DIR / "japan_macro_indicators.csv"
TOKYO_STATIONS_FILE = RAW_DATA_DIR / "tokyo_railway_stations.csv"
PREFECTURE_STATIONS_FILE = RAW_DATA_DIR / "japan_prefecture_railway_stations.csv"

RANDOM_STATE = 42
TARGET = "price"
