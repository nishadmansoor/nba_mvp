from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from nba_mvp.data.io import RAW_DIR, PROCESSED_DIR, read_csv, write_csv
from nba_mvp.features.build_training_set import prepare_features, _normalize_cols
from nba_mvp.modeling.train import MODEL_PATH, predict

DEFAULT_SEASON = "2024-25"


def leaderboard_path(season: str) -> Path:
    return PROCESSED_DIR / f"leaderboard_{season}.csv"


def explain(models: dict, row: pd.DataFrame, baseline: pd.Series) -> pd.Series:
    """
    How much each stat moves this player's predicted vote share: swap the stat
    for a typical rotation player's value and measure the drop. Positive values
    mean the stat is helping the player's case.
    """
    feature_cols = models["feature_cols"]
    base_pred = models["pipeline"].predict(row[feature_cols])[0]
    contribs = {}
    for c in feature_cols:
        swapped = row[feature_cols].copy()
        swapped[c] = baseline[c]
        contribs[c] = base_pred - models["pipeline"].predict(swapped)[0]
    return pd.Series(contribs).sort_values(key=abs, ascending=False)


def rotation_baseline(season_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Median stat line among rotation players (20+ mpg) in the same season."""
    rotation = season_df[season_df["mp"] >= 20]
    return (rotation if len(rotation) else season_df)[feature_cols].median()


def main() -> None:
    p = argparse.ArgumentParser(description="Build an MVP leaderboard for one season.")
    p.add_argument("--season", default=DEFAULT_SEASON, help="season to rank, e.g. 2024-25")
    p.add_argument("--stats", default=str(RAW_DIR / "player_season_stats.csv"),
                   help="player season stats CSV (Basketball-Reference schema)")
    p.add_argument("--standings", default=str(RAW_DIR / "team_standings.csv"))
    p.add_argument("--output", default=None)
    args = p.parse_args()

    models = joblib.load(MODEL_PATH)
    if args.season <= models["train_through"]:
        print(f"Warning: model was trained through {models['train_through']}, "
              f"so {args.season} is in-sample.")

    stats = read_csv(Path(args.stats))
    stats = stats[stats["season"] == args.season]
    if stats.empty:
        raise ValueError(f"No player stats for {args.season} in {args.stats}")
    standings = read_csv(Path(args.standings)) if Path(args.standings).exists() else None

    df = predict(models, prepare_features(stats, standings))

    # Attach actual results when the season's voting is known
    voting_path = RAW_DIR / "mvp_voting.csv"
    if voting_path.exists():
        voting = _normalize_cols(read_csv(voting_path))
        voting = voting[voting["season"] == args.season]
        if not voting.empty:
            df = df.merge(voting[["player", "vote_share", "rank"]].rename(
                columns={"vote_share": "actual_vote_share", "rank": "actual_rank"}),
                on="player", how="left")
            df["actual_vote_share"] = df["actual_vote_share"].fillna(0.0)

    df = df.sort_values("pred_vote_share", ascending=False).reset_index(drop=True)
    df.insert(0, "pred_rank", range(1, len(df) + 1))

    out = Path(args.output) if args.output else leaderboard_path(args.season)
    write_csv(df, out)
    print(f"Wrote leaderboard ({len(df)} players) -> {out}")
    cols = ["pred_rank", "player", "team", "pred_vote_share", "pred_winner_prob", "heuristic_score"]
    cols += [c for c in ["actual_rank", "actual_vote_share"] if c in df.columns]
    print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
