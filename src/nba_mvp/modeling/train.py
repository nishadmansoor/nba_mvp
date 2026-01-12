from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

from nba_mvp.data.io import PROCESSED_DIR, MODELS_DIR, read_csv

MODEL_PATH = MODELS_DIR / "mvp_vote_share_model.joblib"

def time_based_split(df: pd.DataFrame, test_seasons: int = 2):
    seasons = sorted(df["season"].unique())
    test = seasons[-test_seasons:]
    train = seasons[:-test_seasons]
    return df[df["season"].isin(train)].copy(), df[df["season"].isin(test)].copy()

def build_pipeline(feature_cols: list[str], cat_cols: list[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, feature_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=600,
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", pre), ("model", model)])

def main() -> None:
    df = read_csv(PROCESSED_DIR / "training_set.csv")

    target = "vote_share"
    cat_cols = [c for c in ["team"] if c in df.columns]
    ignore = {"season", "player", target, "season_end_year"} | set(cat_cols)

    feature_cols = [c for c in df.columns if c not in ignore]
    X = df[feature_cols + cat_cols].copy()
    y = df[target].astype(float).values

    train_df, test_df = time_based_split(df, test_seasons=2)
    X_train = train_df[feature_cols + cat_cols]
    y_train = train_df[target].astype(float).values
    X_test = test_df[feature_cols + cat_cols]
    y_test = test_df[target].astype(float).values

    pipe = build_pipeline(feature_cols=feature_cols, cat_cols=cat_cols)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "feature_cols": feature_cols, "cat_cols": cat_cols},
        MODEL_PATH
    )
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Test seasons: {sorted(test_df['season'].unique())}")
    print(f"MAE={mae:.4f} RMSE={rmse:.4f}")

if __name__ == "__main__":
    main()
