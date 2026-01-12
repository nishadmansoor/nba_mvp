from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from nba_mvp.data.io import PROCESSED_DIR, MODELS_DIR, read_csv

MODEL_PATH = MODELS_DIR / "mvp_vote_share_model.joblib"

def topk_accuracy(df: pd.DataFrame, k: int) -> float:
    # winner = max vote_share in each season
    seasons = df["season"].unique()
    hits = 0
    for s in seasons:
        sub = df[df["season"] == s].copy()
        true_winner = sub.sort_values("vote_share", ascending=False).iloc[0]["player"]
        topk = sub.sort_values("pred_vote_share", ascending=False).head(k)["player"].tolist()
        hits += int(true_winner in topk)
    return hits / len(seasons)

def main() -> None:
    obj = joblib.load(MODEL_PATH)
    pipe = obj["pipeline"]
    feature_cols = obj["feature_cols"]
    cat_cols = obj["cat_cols"]

    df = read_csv(PROCESSED_DIR / "training_set.csv")

    # Evaluate on last 2 seasons (same as training script split)
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-2:]
    test_df = df[df["season"].isin(test_seasons)].copy()

    X = test_df[feature_cols + cat_cols]
    test_df["pred_vote_share"] = pipe.predict(X)

    # ranking correlation per season
    corrs = []
    for s in test_seasons:
        sub = test_df[test_df["season"] == s].copy()
        # only compare players that received votes to reduce trivial zeros
        sub_votes = sub[sub["vote_share"] > 0].copy()
        if len(sub_votes) >= 5:
            corr, _ = spearmanr(sub_votes["vote_share"], sub_votes["pred_vote_share"])
            corrs.append(corr)

    print(f"Test seasons: {test_seasons}")
    print(f"Top-1 accuracy: {topk_accuracy(test_df, 1):.3f}")
    print(f"Top-3 accuracy: {topk_accuracy(test_df, 3):.3f}")
    if corrs:
        print(f"Avg Spearman (vote-getters only): {float(np.nanmean(corrs)):.3f}")
    else:
        print("Spearman: not enough vote-getter rows in test seasons.")

if __name__ == "__main__":
    main()
