from __future__ import annotations

import argparse
from io import StringIO

import pandas as pd

from nba_mvp.data.bbref import clean_player_names, fetch, flatten_columns, season_str, uncomment_tables
from nba_mvp.data.io import RAW_DIR, write_csv

OUT_PATH = RAW_DIR / "mvp_voting.csv"


def scrape_mvp_voting(ending_year: int) -> pd.DataFrame:
    url = f"https://www.basketball-reference.com/awards/awards_{ending_year}.html"
    html_raw = fetch(url)
    html = uncomment_tables(html_raw)

    # --------------------------------------------------
    # 1) FIND THE MVP TABLE
    # --------------------------------------------------
    mvp = None
    for table_id in ("mvp", "mvp_voting"):
        try:
            mvp = pd.read_html(StringIO(html), attrs={"id": table_id})[0]
            break
        except ValueError:
            pass

    if mvp is None:
        for t in pd.read_html(StringIO(html)):
            cols = [str(b).strip().lower() for (_, b) in t.columns] if isinstance(t.columns, pd.MultiIndex) \
                else [str(c).strip().lower() for c in t.columns]
            if ("share" in cols) and (("player" in cols) or ("name" in cols)):
                mvp = t
                break

    if mvp is None:
        debug_path = RAW_DIR / f"debug_awards_{ending_year}.html"
        debug_path.write_text(html_raw, encoding="utf-8")
        raise RuntimeError(
            f"MVP voting table not found for {ending_year}. "
            f"Saved HTML to {debug_path} for inspection."
        )

    # --------------------------------------------------
    # 2) FLATTEN + NORMALIZE COLUMNS (THIS FIXES 2010)
    # --------------------------------------------------
    mvp = flatten_columns(mvp)

    # --------------------------------------------------
    # 3) VALIDATION + FEATURE CREATION
    # --------------------------------------------------
    if "player" not in mvp.columns:
        if "name" in mvp.columns:
            mvp = mvp.rename(columns={"name": "player"})
        else:
            raise RuntimeError(f"Player column not found for {ending_year}: {mvp.columns.tolist()}")

    if "share" not in mvp.columns:
        raise RuntimeError(f"'share' column not found for {ending_year}: {mvp.columns.tolist()}")

    mvp = mvp[["player", "share"]].rename(columns={"share": "vote_share"})
    mvp["vote_share"] = pd.to_numeric(mvp["vote_share"], errors="coerce")
    mvp = mvp.dropna(subset=["vote_share"])
    mvp["player"] = clean_player_names(mvp["player"])
    mvp["season"] = season_str(ending_year)
    mvp["rank"] = mvp["vote_share"].rank(ascending=False, method="min").astype(int)
    mvp["is_winner"] = (mvp["rank"] == 1).astype(int)

    return mvp[["season", "player", "vote_share", "rank", "is_winner"]]


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape MVP voting from Basketball-Reference.")
    p.add_argument("--start", type=int, default=2010, help="first season ending year")
    p.add_argument("--end", type=int, default=2025, help="last season ending year")
    args = p.parse_args()

    all_rows = []
    for y in range(args.start, args.end + 1):
        print(f"Scraping MVP voting for {season_str(y)}...")
        all_rows.append(scrape_mvp_voting(y))

    out = pd.concat(all_rows, ignore_index=True)
    write_csv(out, OUT_PATH)
    print(
        f"Wrote {OUT_PATH} with {len(out):,} rows across "
        f"{out['season'].nunique()} seasons."
    )


if __name__ == "__main__":
    main()
