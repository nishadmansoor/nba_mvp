from __future__ import annotations

import argparse
import re

import pandas as pd
from bs4 import BeautifulSoup

from nba_mvp.data.bbref import fetch, season_str, uncomment_tables
from nba_mvp.data.io import RAW_DIR, write_csv

OUT_PATH = RAW_DIR / "team_standings.csv"


def scrape_team_standings(ending_year: int) -> pd.DataFrame:
    """
    Team W/L for a season, keyed by the same team abbreviation used in the
    player stats tables (read from each team's link, e.g. /teams/OKC/2025.html).
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{ending_year}_standings.html"
    soup = BeautifulSoup(uncomment_tables(fetch(url)), "lxml")

    def _rows(table_ids: tuple[str, ...]) -> list[dict]:
        out = []
        for table_id in table_ids:
            table = soup.find("table", {"id": table_id})
            if table is None:
                continue
            for tr in table.find("tbody").find_all("tr"):
                link = tr.find("a", href=re.compile(r"/teams/[A-Z]{3}/"))
                wins = tr.find("td", {"data-stat": "wins"})
                losses = tr.find("td", {"data-stat": "losses"})
                if link is None or wins is None or losses is None:
                    continue
                out.append({
                    "team": re.search(r"/teams/([A-Z]{3})/", link["href"]).group(1),
                    "team_name": link.text.strip(),
                    "w": int(wins.text),
                    "l": int(losses.text),
                })
        return out

    # Conference tables cover every team; division tables are a fallback for older layouts
    rows = _rows(("confs_standings_E", "confs_standings_W")) or _rows(("divs_standings_E", "divs_standings_W"))
    if not rows:
        raise RuntimeError(f"No standings found for {ending_year} at {url}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["team"])
    df["win_pct"] = df["w"] / (df["w"] + df["l"])
    df["season"] = season_str(ending_year)
    return df[["season", "team", "team_name", "w", "l", "win_pct"]]


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape team standings from Basketball-Reference.")
    p.add_argument("--start", type=int, default=2010, help="first season ending year")
    p.add_argument("--end", type=int, default=2025, help="last season ending year")
    args = p.parse_args()

    all_rows = []
    for y in range(args.start, args.end + 1):
        print(f"Scraping team standings for {season_str(y)}...")
        all_rows.append(scrape_team_standings(y))

    out = pd.concat(all_rows, ignore_index=True)
    write_csv(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out):,} rows across {out['season'].nunique()} seasons.")


if __name__ == "__main__":
    main()
