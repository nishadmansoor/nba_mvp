from __future__ import annotations

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba-mvp-tracker/1.0)"}

# Basketball-Reference allows ~20 requests/minute; stay under it.
REQUEST_DELAY_S = 3.5


def season_str(ending_year: int) -> str:
    # e.g., 2025 -> "2024-25"
    return f"{ending_year-1}-{str(ending_year)[-2:]}"


def fetch(url: str, retries: int = 3) -> str:
    """GET a Basketball-Reference page (UTF-8), backing off on rate limits."""
    for attempt in range(retries):
        resp = requests.get(url, timeout=30, headers=HEADERS)
        if resp.status_code == 429 and attempt < retries - 1:
            time.sleep(60)
            continue
        resp.raise_for_status()
        # The site doesn't always declare a charset, and requests then falls back
        # to ISO-8859-1, which mangles names like "Jokić".
        resp.encoding = "utf-8"
        time.sleep(REQUEST_DELAY_S)
        return resp.text
    raise RuntimeError(f"Rate limited fetching {url}")


def uncomment_tables(html: str) -> str:
    """Basketball-Reference sometimes wraps tables inside HTML comments."""
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in c:
            c.replace_with(BeautifulSoup(c, "lxml"))
    return str(soup)


def clean_col(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [clean_col(b) for (_, b) in df.columns.to_list()]
    else:
        df.columns = [clean_col(c) for c in df.columns]
    return df


def clean_player_names(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("*", "", regex=False)
        .str.replace(" ", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
