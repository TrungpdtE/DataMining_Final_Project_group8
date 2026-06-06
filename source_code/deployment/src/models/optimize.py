from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from src.config import MODEL_DIR, RANDOM_STATE, REPORT_DIR
from src.models.train_xgb import make_train_test_frames


def optimize(n_iter: int = 25) -> pd.DataFrame:
    train_df, _, y_train, _, artifacts = make_train_test_frames()
    feature_columns = artifacts.numeric_features + artifacts.categorical_features

    model_name = artifacts.model.named_steps["model"].__class__.__name__
    if model_name == "XGBRegressor":
        param_distributions = {
            "model__n_estimators": [500, 700, 900, 1200],
            "model__learning_rate": [0.015, 0.02, 0.03, 0.05],
            "model__max_depth": [4, 5, 6, 7, 8],
            "model__min_child_weight": [1, 3, 5, 8],
            "model__subsample": [0.75, 0.85, 0.95],
            "model__colsample_bytree": [0.75, 0.85, 0.95],
            "model__reg_alpha": [0, 0.03, 0.05, 0.1],
            "model__reg_lambda": [1, 1.5, 2, 3],
        }
    else:
        param_distributions = {
            "model__learning_rate": [0.02, 0.03, 0.04, 0.06],
            "model__max_iter": [400, 700, 1000],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0, 0.03, 0.05, 0.1],
        }
    search = RandomizedSearchCV(
        artifacts.model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(train_df[feature_columns], y_train)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results.to_csv(REPORT_DIR / "xgb_random_search_results.csv", index=False)

    joblib.dump(
        {
            "model": search.best_estimator_,
            "clusterer": artifacts.clusterer,
            "target_encoder": artifacts.target_encoder,
            "feature_columns": feature_columns,
            "best_params": search.best_params_,
        },
        MODEL_DIR / "xgb_pipeline_optimized.pkl",
    )
    print("Best RMSE:", -search.best_score_)
    print("Best params:", search.best_params_)
    return results


if __name__ == "__main__":
    optimize()
