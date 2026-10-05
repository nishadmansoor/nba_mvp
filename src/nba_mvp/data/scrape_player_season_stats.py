from __future__ import annotations

import argparse
from io import StringIO

import pandas as pd

from nba_mvp.data.bbref import clean_player_names, fetch, flatten_columns, season_str, uncomment_tables
from nba_mvp.data.io import RAW_DIR, write_csv

OUT_PATH = RAW_DIR / "player_season_stats.csv"


def _read_table_by_ids_or_scan(html: str, ids: list[str], must_have: list[str], debug_name: str) -> pd.DataFrame:
    """
    Try to read a table by several possible ids; if that fails, scan all tables
    and return the first table whose columns contain all tokens in must_have.
    Saves debug HTML on failure.
    """
    # 1) Try known ids first
    for table_id in ids:
        try:
            return pd.read_html(StringIO(html), attrs={"id": table_id})[0]
        except ValueError:
            pass

    # 2) Fallback: scan all tables
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        tables = []

    for t in tables:
        cols = list(flatten_columns(t).columns)
        if all(token in cols for token in must_have):
            return t

    # 3) Save debug HTML for inspection
    debug_path = RAW_DIR / f"debug_{debug_name}.html"
    debug_path.write_text(html, encoding="utf-8")
    raise RuntimeError(f"Could not find expected table. Saved HTML to {debug_path}")


def _scrape_table(url: str, ids: list[str], must_have: list[str], debug_name: str) -> pd.DataFrame:
    html = uncomment_tables(fetch(url))
    df = flatten_columns(_read_table_by_ids_or_scan(html, ids, must_have, debug_name))

    # Older pages use "Tm"; newer ones use "Team"
    if "tm" in df.columns and "team" not in df.columns:
        df = df.rename(columns={"tm": "team"})
    if "player" not in df.columns:
        raise RuntimeError(f"Table at {url} missing 'player': {df.columns.tolist()}")

    # Remove repeated header rows and the "League Average" footer
    if "rk" in df.columns:
        df = df[df["rk"].astype(str) != "Rk"]
    df = df[df["player"].notna() & (df["player"] != "League Average")].copy()
    df["player"] = clean_player_names(df["player"])
    df["team"] = df["team"].astype(str)
    return df


def _is_total_row(team: pd.Series) -> pd.Series:
    # Traded players get a season-total row: "TOT" on older pages, "2TM"/"3TM" on newer ones
    return (team == "TOT") | team.str.fullmatch(r"\d+TM")


def _collapse_traded(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per player (the season total for traded players) and record
    each stint as "TEAM:games|TEAM:games" so team win% can be games-weighted.
    """
    total = _is_total_row(df["team"])
    stints = df[~total]
    teams = (
        stints.assign(_s=stints["team"] + ":" + pd.to_numeric(stints["g"], errors="coerce").fillna(0).astype(int).astype(str))
        .groupby("player", sort=False)["_s"]
        .agg("|".join)
        .rename("teams")
    )
    df = df.assign(_tot=(~total).astype(int)).sort_values(["player", "_tot"], kind="stable")
    df = df.drop_duplicates(subset=["player"], keep="first").drop(columns=["_tot"])
    return df.merge(teams, on="player", how="left")


def scrape_player_stats_for_year(ending_year: int) -> pd.DataFrame:
    """
    Pull per-game + advanced player stats for a given season ending year
    and return a merged dataframe keyed by player.
    """
    per_game = _scrape_table(
        f"https://www.basketball-reference.com/leagues/NBA_{ending_year}_per_game.html",
        ids=["per_game_stats", "per_game", "all_per_game_stats"],
        must_have=["player", "pts"],
        debug_name=f"per_game_{ending_year}",
    )
    adv = _scrape_table(
        f"https://www.basketball-reference.com/leagues/NBA_{ending_year}_advanced.html",
        ids=["advanced", "advanced_stats", "all_advanced_stats"],
        must_have=["player", "ws"],  # advanced pages include WS
        debug_name=f"advanced_{ending_year}",
    )

    per_game = _collapse_traded(per_game)
    adv = _collapse_traded(adv)

    # Advanced "mp" is season total minutes; per-game "mp" is minutes per game
    adv = adv.rename(columns={"mp": "mp_total"})
    shared = [c for c in adv.columns if c in per_game.columns and c != "player"]
    merged = per_game.merge(adv.drop(columns=shared), on="player", how="inner")
    merged["season"] = season_str(ending_year)

    return merged


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape per-game + advanced player stats from Basketball-Reference.")
    p.add_argument("--start", type=int, default=2010, help="first season ending year")
    p.add_argument("--end", type=int, default=2025, help="last season ending year")
    args = p.parse_args()

    all_rows: list[pd.DataFrame] = []
    for y in range(args.start, args.end + 1):
        print(f"Scraping player stats for {season_str(y)}...")
        all_rows.append(scrape_player_stats_for_year(y))

    out = pd.concat(all_rows, ignore_index=True)
    write_csv(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out):,} rows across {out['season'].nunique()} seasons.")


if __name__ == "__main__":
    main()
