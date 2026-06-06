from __future__ import annotations

import json
import os
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

from src.config import (
    FIGURE_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
)
from src.features.feature_engineering import (
    LocationClusterer,
    TargetMeanEncoder,
    prepare_model_frame,
)
from src.preprocessing.clean_data import load_clean_merged


NUMERIC_FEATURES = [
    "area",
    "floor_area",
    "frontage",
    "house_age",
    "coverage",
    "FloorAreaRatio",
    "Year",
    "Quarter",
    "Migration",
    "AverageTimeToStation",
    "MunicipalityCategory",
    "density",
    "latitude",
    "longitude",
    "Distance_to_designated_city",
    "dist_to_tokyo_km",
    "dist_to_osaka_km",
    "dist_to_nagoya_km",
    "dist_to_fukuoka_km",
    "dist_to_sapporo_km",
    "dist_to_nearest_major_center_km",
    "log_dist_to_tokyo",
    "log_dist_to_nearest_major_center",
    "quarter_sin",
    "quarter_cos",
    "location_cluster",
    "Prefecture_target_mean",
    "Location_target_mean",
    "Close_to_Tokyo",
    "Close_to_greater_Tokyo_area",
    "Close_to_designated_city_flag",
    "RegionCommercialArea",
    "RegionIndustrialArea",
    "RegionPotentialResidentialArea",
    "RegionResidentialArea",
    "Region_Chubu",
    "Region_Chugoku",
    "Region_Hokkaido",
    "Region_Kansai",
    "Region_Kanto",
    "Region_Kyushu",
    "Region_Shikoku",
    "Region_Tohoku",
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
]

CATEGORICAL_FEATURES = [
    "Type",
    "Prefecture",
    "nearest_major_center",
    "Nearest_designated_city",
]


@dataclass
class TrainingArtifacts:
    model: Pipeline
    clusterer: LocationClusterer
    target_encoder: TargetMeanEncoder
    numeric_features: list[str]
    categorical_features: list[str]


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def make_regressor(model_name: str = "auto"):
    model_name = model_name.lower()
    if model_name == "auto":
        model_name = "xgboost" if XGBRegressor is not None else "hist_gbdt"

    if model_name in {"xgboost", "xgb"}:
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed")
        return XGBRegressor(
            n_estimators=900,
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.5,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name in {"lightgbm", "lgbm"}:
        if LGBMRegressor is None:
            raise ImportError("lightgbm is not installed")
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.025,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.5,
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name in {"extra_trees", "extratrees"}:
        return ExtraTreesRegressor(
            n_estimators=260,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if model_name in {"mlp", "mlp_regressor"}:
        return MLPRegressor(
            hidden_layer_sizes=(192, 96, 48),
            activation="relu",
            alpha=0.0007,
            learning_rate_init=0.001,
            batch_size=256,
            early_stopping=True,
            validation_fraction=0.12,
            max_iter=140,
            random_state=RANDOM_STATE,
        )
    if model_name in {"hist_gbdt", "hist_gradient_boosting", "sklearn_gbdt"}:
        return HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=450,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_model(model_name: str = "auto") -> Pipeline:
    regressor = make_regressor(model_name)
    numeric_transformer = (
        StandardScaler()
        if isinstance(regressor, MLPRegressor)
        else "passthrough"
    )
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, _existing_columns_placeholder()),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            _existing_columns_placeholder(),
                        ),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", regressor),
        ]
    )


def _existing_columns_placeholder() -> list[str]:
    return []


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str = "auto",
) -> Pipeline:
    pipeline = make_model(model_name)
    numeric_transformer = pipeline.named_steps["preprocess"].transformers[0][1]
    pipeline.set_params(
        preprocess__transformers=[
            ("num", numeric_transformer, numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    return pipeline


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model: Pipeline, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return pd.DataFrame(
        [
            {
                "model": model.named_steps["model"].__class__.__name__,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "train_year_min": int(X_train["Year"].min()) if "Year" in X_train else None,
                "train_year_max": int(X_train["Year"].max()) if "Year" in X_train else None,
                "test_year_min": int(X_test["Year"].min()) if "Year" in X_test else None,
                "test_year_max": int(X_test["Year"].max()) if "Year" in X_test else None,
                "rmse_train_log": rmse(y_train, train_pred),
                "rmse_test_log": rmse(y_test, test_pred),
                "mae_test_log": float(mean_absolute_error(y_test, test_pred)),
                "r2_test_log": float(r2_score(y_test, test_pred)),
                "mae_test_yen": float(
                    mean_absolute_error(np.expm1(y_test), np.expm1(test_pred))
                ),
            }
        ]
    )


def save_eda_figures(before: pd.DataFrame, after: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.histplot(np.log1p(before[TARGET]), kde=True, bins=50)
    plt.title("Log Price Distribution Before Outlier Removal")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "price_log_before_outlier.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(np.log1p(after[TARGET]), kde=True, bins=50)
    plt.title("Log Price Distribution After Outlier Removal")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "price_log_after_outlier.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=before["area"])
    plt.title("Area Before")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=after["area"])
    plt.title("Area After")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "area_outlier_before_after.png", dpi=160)
    plt.close()

    missing = before.isna().mean().sort_values(ascending=False).head(20) * 100
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing.values, y=missing.index)
    plt.xlabel("Missing (%)")
    plt.title("Top Missing Columns Before Cleaning")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "missing_before_cleaning.png", dpi=160)
    plt.close()


def save_model_figures(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.25)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red")
    plt.xlabel("Actual log price")
    plt.ylabel("Predicted log price")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residual_distribution.png", dpi=160)
    plt.close()

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    regressor = model.named_steps["model"]
    if hasattr(regressor, "feature_importances_"):
        importances = pd.Series(
            regressor.feature_importances_, index=feature_names
        ).sort_values(ascending=False)
    else:
        from sklearn.inspection import permutation_importance

        sample_size = min(1000, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
        y_sample = y_test.loc[X_sample.index]
        perm = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
        )
        importances = pd.Series(
            perm.importances_mean, index=X_test.columns
        ).sort_values(ascending=False)
    importances.head(20).to_csv(REPORT_DIR / "feature_importance.csv")
    plt.figure(figsize=(10, 7))
    sns.barplot(x=importances.head(15).values, y=importances.head(15).index)
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=160)
    plt.close()


def _configured_max_rows() -> int:
    value = os.getenv("JAPAN_HOUSE_MAX_ROWS", "50000").strip()
    try:
        return int(value)
    except ValueError:
        return 50000


def _configured_test_year(df: pd.DataFrame) -> int:
    value = os.getenv("JAPAN_HOUSE_TEST_YEAR", "").strip()
    years = sorted(pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).unique())
    if value:
        year = int(value)
        if year in years:
            return year
    return years[-1]


def _configured_model_names() -> list[str]:
    value = os.getenv("JAPAN_HOUSE_MODELS", "auto,extra_trees,mlp").strip()
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names or ["auto"]


def _add_missing_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defaults = {
        "gdp_growth": 0.0,
        "interest_rate": 0.0,
        "inflation_rate": 0.0,
        "population_growth": 0.0,
        "housing_price_index": 100.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def make_train_test_frames(
    max_rows: int | None = None,
    model_name: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, TrainingArtifacts]:
    before_clean, df = load_clean_merged(drop_outliers=True)
    df = prepare_model_frame(df)
    df = _add_missing_macro_features(df)
    if max_rows is None:
        max_rows = _configured_max_rows()
    if max_rows and len(df) > max_rows:
        sampled_parts = []
        for _, part in df.groupby("Year"):
            n = min(
                len(part),
                max(1, round(max_rows * len(part) / len(df))),
            )
            sampled_parts.append(part.sample(n, random_state=RANDOM_STATE))
        df = pd.concat(sampled_parts, ignore_index=True)
    test_year = _configured_test_year(df)
    train_df = df[df["Year"] < test_year].copy()
    test_df = df[df["Year"] == test_year].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Cannot temporal split with holdout year {test_year}")
    y_train = np.log1p(train_df[TARGET])
    y_test = np.log1p(test_df[TARGET])

    clusterer = LocationClusterer(n_clusters=12)
    clusterer.fit(train_df)
    train_df = clusterer.transform(train_df)
    test_df = clusterer.transform(test_df)

    target_encoder = TargetMeanEncoder(columns=["Prefecture", "Location"], smoothing=40)
    target_encoder.fit(train_df, y_train)
    train_df = target_encoder.transform(train_df)
    test_df = target_encoder.transform(test_df)

    numeric_features = _existing_columns(train_df, NUMERIC_FEATURES)
    categorical_features = _existing_columns(train_df, CATEGORICAL_FEATURES)
    model = build_pipeline(numeric_features, categorical_features, model_name=model_name)
    artifacts = TrainingArtifacts(
        model=model,
        clusterer=clusterer,
        target_encoder=target_encoder,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    save_eda_figures(before_clean, df)
    return train_df, test_df, y_train, y_test, artifacts


def train() -> pd.DataFrame:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    best = None
    skipped = []
    for model_name in _configured_model_names():
        try:
            train_df, test_df, y_train, y_test, artifacts = make_train_test_frames(
                model_name=model_name
            )
            feature_columns = artifacts.numeric_features + artifacts.categorical_features
            artifacts.model.fit(train_df[feature_columns], y_train)
            metrics = evaluate(
                artifacts.model,
                train_df[feature_columns],
                y_train,
                test_df[feature_columns],
                y_test,
            )
            metrics["requested_model"] = model_name
            all_metrics.append(metrics)
            score = float(metrics["rmse_test_log"].iloc[0])
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "artifacts": artifacts,
                    "feature_columns": feature_columns,
                    "train_df": train_df,
                    "test_df": test_df,
                    "y_train": y_train,
                    "y_test": y_test,
                    "metrics": metrics,
                    "requested_model": model_name,
                }
        except Exception as exc:
            skipped.append({"requested_model": model_name, "error": str(exc)})

    if best is None:
        raise RuntimeError(f"No model could be trained. Skipped: {skipped}")

    metrics = pd.concat(all_metrics, ignore_index=True).sort_values("rmse_test_log")
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(REPORT_DIR / "skipped_models.csv", index=False)

    train_out = best["train_df"][best["feature_columns"]].copy()
    train_out[TARGET] = best["y_train"].values
    test_out = best["test_df"][best["feature_columns"]].copy()
    test_out[TARGET] = best["y_test"].values
    train_out.to_csv(PROCESSED_DATA_DIR / "train_processed.csv", index=False)
    test_out.to_csv(PROCESSED_DATA_DIR / "test_processed.csv", index=False)

    save_model_figures(best["artifacts"].model, best["test_df"][best["feature_columns"]], best["y_test"])

    bundle = {
        "model": best["artifacts"].model,
        "clusterer": best["artifacts"].clusterer,
        "target_encoder": best["artifacts"].target_encoder,
        "feature_columns": best["feature_columns"],
        "numeric_features": best["artifacts"].numeric_features,
        "categorical_features": best["artifacts"].categorical_features,
        "holdout_year": int(best["test_df"]["Year"].min()),
        "requested_model": best["requested_model"],
    }
    joblib.dump(bundle, MODEL_DIR / "xgb_pipeline.pkl")
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(
            {
                "target": "log1p(price)",
                "model": best["artifacts"].model.named_steps["model"].__class__.__name__,
                "requested_model": best["requested_model"],
                "holdout_year": int(best["test_df"]["Year"].min()),
                "split_strategy": "temporal_holdout_train_years_before_test_year",
                "feature_columns": best["feature_columns"],
                "candidate_models": _configured_model_names(),
                "skipped_models": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    train()
