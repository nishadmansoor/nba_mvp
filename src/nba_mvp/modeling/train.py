from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_mvp.data.io import PROCESSED_DIR, MODELS_DIR, read_csv
from nba_mvp.features.build_training_set import FEATURE_COLS

MODEL_PATH = MODELS_DIR / "mvp_vote_share_model.joblib"

# Predicting 2024-25, so the model only learns from seasons before it
DEFAULT_TRAIN_THROUGH = "2023-24"


def time_based_split(df: pd.DataFrame, test_seasons: int = 2):
    seasons = sorted(df["season"].unique())
    test = seasons[-test_seasons:]
    train = seasons[:-test_seasons]
    return df[df["season"].isin(train)].copy(), df[df["season"].isin(test)].copy()


def build_regressor() -> Pipeline:
    """Primary model: predicts each player's MVP vote share."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            max_iter=300,
            min_samples_leaf=20,
            random_state=42,
        )),
    ])


def build_classifier() -> Pipeline:
    """Secondary model: is this player the season's MVP? (one positive per season)"""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
    ])


def fit_models(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> dict:
    # The regressor learns from every player (the zeros help it rank vote-getters);
    # the classifier only compares plausible candidates (see is_candidate).
    regressor = build_regressor().fit(df[feature_cols], df["vote_share"].astype(float).values)
    cand = df[df["is_candidate"] == 1]
    classifier = build_classifier().fit(cand[feature_cols], cand["is_winner"].astype(int).values)
    return {"pipeline": regressor, "classifier": classifier, "feature_cols": feature_cols}


def predict(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add pred_vote_share and pred_winner_prob (sums to 1 within each season)."""
    out = df.copy()
    X = out[models["feature_cols"]]
    candidate = out["is_candidate"] == 1
    out["pred_vote_share"] = np.where(candidate, np.clip(models["pipeline"].predict(X), 0.0, 1.0), 0.0)
    raw = pd.Series(np.where(candidate, models["classifier"].predict_proba(X)[:, 1], 0.0), index=out.index)
    out["pred_winner_prob"] = raw / raw.groupby(out["season"]).transform("sum")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train MVP vote-share + winner models.")
    p.add_argument("--train-through", default=DEFAULT_TRAIN_THROUGH,
                   help="last season (e.g. 2023-24) the final model may learn from")
    args = p.parse_args()

    df = read_csv(PROCESSED_DIR / "training_set.csv")
    df = df[df["season"] <= args.train_through]
    if df.empty:
        raise ValueError(f"No training seasons on or before {args.train_through}")

    # Holdout check: train on older seasons, score the two most recent
    train_df, test_df = time_based_split(df, test_seasons=2)
    holdout = predict(fit_models(train_df), test_df)
    mae = mean_absolute_error(holdout["vote_share"], holdout["pred_vote_share"])
    rmse = root_mean_squared_error(holdout["vote_share"], holdout["pred_vote_share"])

    # Final model learns from every season up to --train-through
    models = fit_models(df)
    models["train_through"] = args.train_through
    models["train_seasons"] = sorted(df["season"].unique())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Trained on {len(models['train_seasons'])} seasons: "
          f"{models['train_seasons'][0]} .. {models['train_seasons'][-1]}")
    print(f"Holdout seasons: {sorted(test_df['season'].unique())}  MAE={mae:.4f} RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
