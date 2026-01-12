from __future__ import annotations

from pathlib import Path
import pandas as pd

from nba_mvp.data.io import RAW_DIR, PROCESSED_DIR, read_csv, write_csv

FEATURE_COLS_DEFAULT = [
    "g", "mp", "pts", "trb", "ast", "stl", "blk",
    "ws", "ws_per_48",
    # optional columns if you add them:
    "bpm", "vorp", "ts_pct",
    "win_pct",
]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # common aliases
    aliases = {
        "season": "season",
        "player": "player",
        "tm": "team",
        "team": "team",
        "g": "g",
        "mp": "mp",
        "pts": "pts",
        "trb": "trb",
        "reb": "trb",
        "ast": "ast",
        "stl": "stl",
        "blk": "blk",
        "ws": "ws",
        "ws/48": "ws_per_48",
        "ws_per_48": "ws_per_48",
        "bpm": "bpm",
        "vorp": "vorp",
        "ts%": "ts_pct",
        "ts_pct": "ts_pct",
        "win_pct": "win_pct",
        "win%": "win_pct",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    return df

def build_training_set(
    mvp_voting_path: Path = RAW_DIR / "mvp_voting.csv",
    season_stats_path: Path = RAW_DIR / "player_season_stats.csv",
) -> pd.DataFrame:
    voting = _normalize_cols(read_csv(mvp_voting_path))
    stats = _normalize_cols(read_csv(season_stats_path))

    required_voting = {"season", "player", "vote_share"}
    missing = required_voting - set(voting.columns)
    if missing:
        raise ValueError(f"mvp_voting missing columns: {missing}")

    required_stats = {"season", "player"}
    missing = required_stats - set(stats.columns)
    if missing:
        raise ValueError(f"player_season_stats missing columns: {missing}")

    # merge labels onto features; keep all stats rows (players) but label only where voting exists
    df = stats.merge(voting[["season", "player", "vote_share", "rank"]].drop_duplicates(),
                     on=["season", "player"], how="left")

    # fill non-vote players with 0 share (they received no votes)
    df["vote_share"] = df["vote_share"].fillna(0.0)

    # keep a standard set of features that exist
    feature_cols = [c for c in FEATURE_COLS_DEFAULT if c in df.columns]
    keep = ["season", "player", "team", "vote_share"] + feature_cols
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # basic cleanup
    df["season_end_year"] = df["season"].str[-2:].astype(int) + 2000
    # numeric coercion
    for c in feature_cols + ["vote_share"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + ["vote_share"])

    return df

def main() -> None:
    out_path = PROCESSED_DIR / "training_set.csv"
    df = build_training_set()
    write_csv(df, out_path)
    print(f"Wrote {len(df):,} rows -> {out_path}")

if __name__ == "__main__":
    main()
