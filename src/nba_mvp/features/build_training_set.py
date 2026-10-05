from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_mvp.data.io import RAW_DIR, PROCESSED_DIR, read_csv, write_csv

# Season-level per-game + advanced stats, plus team success
BASE_FEATURES = [
    "g", "mp", "pts", "trb", "ast", "stl", "blk",
    "ws", "ws_per_48", "bpm", "vorp", "ts_pct", "per", "usg_pct",
    "win_pct",
]

# MVP voting is relative: what matters is how a player stacks up against that
# season's league, so these are also expressed as within-season percentiles.
RELATIVE_FEATURES = ["pts", "ast", "trb", "ws", "bpm", "vorp", "per", "win_pct"]

FEATURE_COLS = BASE_FEATURES + [f"{c}_season_pct" for c in RELATIVE_FEATURES]

DISPLAY_COLS = ["season", "player", "team", "pos"]


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # common aliases (including the *_pg/*_adv suffixes from older scrapes)
    aliases = {
        "tm": "team",
        "team_pg": "team",
        "pos_pg": "pos",
        "g_pg": "g",
        "mp_pg": "mp",
        "reb": "trb",
        "ws/48": "ws_per_48",
        "ts%": "ts_pct",
        "usg%": "usg_pct",
        "win%": "win_pct",
        "share": "vote_share",
    }
    return df.rename(columns={c: aliases.get(c, c) for c in df.columns})


def heuristic_score(df: pd.DataFrame) -> pd.Series:
    """Baseline "media narrative" score carried over from the original nba_mvp.py."""
    return (
        df["pts"] * 0.35
        + df["ast"] * 0.20
        + df["trb"] * 0.15
        + df["ws"] * 0.20
        + df["bpm"] * 0.10
    )


def _team_win_pct(stats: pd.DataFrame, standings: pd.DataFrame) -> pd.Series:
    """Games-weighted team win% across every team a player suited up for."""
    lookup = standings.set_index(["season", "team"])["win_pct"].to_dict()

    def _one(season: str, team: str, teams: object) -> float:
        stints = str(teams).split("|") if isinstance(teams, str) and teams else [f"{team}:1"]
        num = den = 0.0
        for stint in stints:
            abbr, _, games = stint.partition(":")
            pct = lookup.get((season, abbr))
            g = float(games or 1)
            if pct is not None and g > 0:
                num += pct * g
                den += g
        return num / den if den else np.nan

    teams = stats["teams"] if "teams" in stats.columns else pd.Series(None, index=stats.index)
    return pd.Series(
        [_one(s, t, ts) for s, t, ts in zip(stats["season"], stats["team"], teams)],
        index=stats.index,
    )


def prepare_features(stats: pd.DataFrame, standings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn raw player season stats into one model-ready row per player-season."""
    df = _normalize_cols(stats)
    df = df[df["player"].notna() & (df["player"] != "League Average")].copy()

    if "win_pct" not in df.columns:
        if standings is None:
            raise ValueError("Need team standings (data/raw/team_standings.csv) to compute win_pct")
        df["win_pct"] = _team_win_pct(df, _normalize_cols(standings))

    for c in BASE_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in RELATIVE_FEATURES:
        df[f"{c}_season_pct"] = df.groupby("season")[c].rank(pct=True)

    df["heuristic_score"] = heuristic_score(df)
    df["season_end_year"] = df["season"].str[-2:].astype(int) + 2000

    # Every vote-getter since 2010 played 59%+ of the season at 25+ mpg; looser
    # thresholds here just keep fringe players from adding noise to the models.
    season_games = df.groupby("season")["g"].transform("max")
    df["is_candidate"] = ((df["g"] >= 0.4 * season_games) & (df["mp"] >= 20)).astype(int)

    for c in DISPLAY_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Need the core box-score stats; anything else is imputed by the model
    df = df.dropna(subset=["pts", "trb", "ast"])
    return df[DISPLAY_COLS + ["season_end_year"] + FEATURE_COLS + ["heuristic_score", "is_candidate"]].reset_index(drop=True)


def build_training_set(
    mvp_voting_path: Path = RAW_DIR / "mvp_voting.csv",
    season_stats_path: Path = RAW_DIR / "player_season_stats.csv",
    standings_path: Path = RAW_DIR / "team_standings.csv",
) -> pd.DataFrame:
    voting = _normalize_cols(read_csv(mvp_voting_path))
    stats = read_csv(season_stats_path)
    standings = read_csv(standings_path) if standings_path.exists() else None

    required_voting = {"season", "player", "vote_share"}
    missing = required_voting - set(voting.columns)
    if missing:
        raise ValueError(f"mvp_voting missing columns: {missing}")

    df = prepare_features(stats, standings)

    # Only train on seasons we actually have voting for
    df = df[df["season"].isin(voting["season"].unique())]

    # merge labels onto features; players who received no votes get a 0 share
    label_cols = [c for c in ["season", "player", "vote_share", "rank"] if c in voting.columns]
    df = df.merge(voting[label_cols].drop_duplicates(subset=["season", "player"]),
                  on=["season", "player"], how="left")
    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce").fillna(0.0)
    df["is_winner"] = (df["vote_share"] == df.groupby("season")["vote_share"].transform("max")).astype(int)

    unmatched = set(map(tuple, voting[["season", "player"]].values)) - set(map(tuple, df[["season", "player"]].values))
    if unmatched:
        print(f"Warning: {len(unmatched)} vote-getters had no matching stats row: {sorted(unmatched)[:5]}")

    return df


def main() -> None:
    out_path = PROCESSED_DIR / "training_set.csv"
    df = build_training_set()
    write_csv(df, out_path)
    print(f"Wrote {len(df):,} rows across {df['season'].nunique()} seasons -> {out_path}")


if __name__ == "__main__":
    main()
