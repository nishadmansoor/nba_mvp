from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from nba_mvp.data.io import RAW_DIR, PROCESSED_DIR, MODELS_DIR, write_csv

MODEL_PATH = MODELS_DIR / "mvp_vote_share_model.joblib"

def _normalize_current(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    aliases = {
        "player_name": "player",
        "team_name": "team",
        "g": "g",
        "gp": "g",
        "min": "mp",
        "pts": "pts",
        "reb": "trb",
        "ast": "ast",
        "stl": "stl",
        "blk": "blk",
        "ws": "ws",
        "ws/48": "ws_per_48",
        "win_pct": "win_pct",
        "win_pct": "win_pct",
        "w": "w",
        "l": "l",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    # If win_pct isn't present but W/L are, compute
    if "win_pct" not in df.columns and "w" in df.columns and "l" in df.columns:
        df["win_pct"] = df["w"] / (df["w"] + df["l"])
    return df

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=str(RAW_DIR / "latest_mvp_predictions.csv"))
    p.add_argument("--output", type=str, default=str(PROCESSED_DIR / "predictions_current.csv"))
    args = p.parse_args()

    obj = joblib.load(MODEL_PATH)
    pipe = obj["pipeline"]
    feature_cols = obj["feature_cols"]
    cat_cols = obj["cat_cols"]

    df = pd.read_csv(args.input)
    df = _normalize_current(df)

    # Keep one row per player (your file already is); if duplicates exist, take latest game_date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values("game_date").drop_duplicates(subset=["player"], keep="last")

    # Ensure required columns exist (fill missing with NA -> pipeline imputers handle)
    for c in feature_cols + cat_cols:
        if c not in df.columns:
            df[c] = pd.NA

    X = df[feature_cols + cat_cols]
    df["pred_vote_share"] = pipe.predict(X)

    # Pretty output
    out = df[["player", "team", "pred_vote_share"] + [c for c in feature_cols if c in df.columns] + ["win_pct"]].copy() if "win_pct" in df.columns else df[["player","team","pred_vote_share"] + [c for c in feature_cols if c in df.columns]].copy()
    out = out.sort_values("pred_vote_share", ascending=False)
    write_csv(out, Path(args.output))
    print(f"Wrote leaderboard -> {args.output}")

if __name__ == "__main__":
    main()
