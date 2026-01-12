from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

from nba_mvp.data.io import RAW_DIR, write_csv

BR_BASE = "https://www.basketball-reference.com"

def _season_url(season_end_year: int) -> str:
    # Example: 2024 -> .../awards/awards_2024.html
    return f"{BR_BASE}/awards/awards_{season_end_year}.html"

def _parse_mvp_table(html: str, season_label: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    # Basketball-Reference often wraps tables in comments; BeautifulSoup doesn't parse those as DOM tables.
    # We fall back to a regex-based extraction of the table HTML if needed.
    table = soup.find("table", {"id": "mvp"})
    if table is None:
        comments = soup.find_all(string=lambda t: isinstance(t, type(soup.comment)))
        mvp_html = None
        for c in comments:
            if "table" in c and 'id="mvp"' in c:
                mvp_html = c
                break
        if mvp_html is None:
            raise RuntimeError("Could not find MVP table on page.")
        table_soup = BeautifulSoup(mvp_html, "lxml")
        table = table_soup.find("table", {"id": "mvp"})
        if table is None:
            raise RuntimeError("Could not parse MVP table from commented HTML.")

    df = pd.read_html(str(table))[0]

    # Common columns include: Rank, Player, Age, Tm, First, Pts Won, Pts Max, Share, etc.
    # Normalize
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in {"rk", "rank"}:
            rename[c] = "rank"
        elif lc == "player":
            rename[c] = "player"
        elif lc in {"tm", "team"}:
            rename[c] = "team"
        elif lc == "share":
            rename[c] = "vote_share"
    df = df.rename(columns=rename)

    keep = [c for c in ["rank", "player", "team", "vote_share"] if c in df.columns]
    df = df[keep].copy()
    df["season"] = season_label

    # Coerce types
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    if "vote_share" in df.columns:
        df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce")

    df = df.dropna(subset=["player"])
    return df

def scrape_mvp_voting(start_season_end_year: int = 1980, end_season_end_year: int = 2025) -> pd.DataFrame:
    rows = []
    for yr in range(start_season_end_year, end_season_end_year + 1):
        season_label = f"{yr-1}-{str(yr)[-2:]}"
        url = _season_url(yr)
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = _parse_mvp_table(r.text, season_label)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    return out

def main() -> None:
    out_path = RAW_DIR / "mvp_voting.csv"
    df = scrape_mvp_voting()
    write_csv(df, out_path)
    print(f"Wrote {len(df):,} rows -> {out_path}")

if __name__ == "__main__":
    main()
