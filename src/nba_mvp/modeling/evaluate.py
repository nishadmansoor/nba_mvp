from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from nba_mvp.data.io import PROCESSED_DIR, REPORTS_DIR, read_csv, write_csv
from nba_mvp.modeling.train import MODEL_PATH, fit_models, predict

BACKTEST_PATH = REPORTS_DIR / "backtest.csv"


def topk_hit(sub: pd.DataFrame, score_col: str, k: int) -> int:
    true_winner = sub.sort_values("vote_share", ascending=False).iloc[0]["player"]
    topk = sub.sort_values(score_col, ascending=False).head(k)["player"].tolist()
    return int(true_winner in topk)


def topk_accuracy(df: pd.DataFrame, k: int, score_col: str = "pred_vote_share") -> float:
    seasons = df["season"].unique()
    return sum(topk_hit(df[df["season"] == s], score_col, k) for s in seasons) / len(seasons)


def score_season(sub: pd.DataFrame) -> dict:
    # only compare players that received votes to reduce trivial zeros
    votes = sub[sub["vote_share"] > 0]
    corr = spearmanr(votes["vote_share"], votes["pred_vote_share"])[0] if len(votes) >= 5 else np.nan
    return {
        "season": sub["season"].iloc[0],
        "actual_winner": sub.sort_values("vote_share", ascending=False).iloc[0]["player"],
        "predicted_winner": sub.sort_values("pred_vote_share", ascending=False).iloc[0]["player"],
        "top1": topk_hit(sub, "pred_vote_share", 1),
        "top3": topk_hit(sub, "pred_vote_share", 3),
        "classifier_top1": topk_hit(sub, "pred_winner_prob", 1),
        "heuristic_top1": topk_hit(sub, "heuristic_score", 1),
        "spearman_vote_getters": corr,
        "mae": mean_absolute_error(sub["vote_share"], sub["pred_vote_share"]),
        "rmse": root_mean_squared_error(sub["vote_share"], sub["pred_vote_share"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward backtest of the MVP models.")
    p.add_argument("--seasons", type=int, default=8, help="number of most recent seasons to backtest")
    args = p.parse_args()

    train_through = joblib.load(MODEL_PATH)["train_through"]
    df = read_csv(PROCESSED_DIR / "training_set.csv")
    df = df[df["season"] <= train_through]

    # Walk forward: for each test season, train only on the seasons before it (no leakage)
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-args.seasons:]
    rows = []
    for s in test_seasons:
        models = fit_models(df[df["season"] < s])
        rows.append(score_season(predict(models, df[df["season"] == s])))
    results = pd.DataFrame(rows)
    write_csv(results, BACKTEST_PATH)

    print(f"Walk-forward backtest over {len(results)} seasons ({test_seasons[0]} .. {test_seasons[-1]}):")
    print(results[["season", "actual_winner", "predicted_winner", "top1", "top3", "spearman_vote_getters"]]
          .to_string(index=False))
    print()
    print(f"Top-1 accuracy (vote share model): {results['top1'].mean():.3f}")
    print(f"Top-3 accuracy (vote share model): {results['top3'].mean():.3f}")
    print(f"Top-1 accuracy (winner classifier): {results['classifier_top1'].mean():.3f}")
    print(f"Top-1 accuracy (heuristic baseline): {results['heuristic_top1'].mean():.3f}")
    print(f"Avg Spearman (vote-getters only): {results['spearman_vote_getters'].mean():.3f}")
    print(f"MAE={results['mae'].mean():.4f} RMSE={results['rmse'].mean():.4f}")
    print(f"Saved -> {BACKTEST_PATH}")


if __name__ == "__main__":
    main()
