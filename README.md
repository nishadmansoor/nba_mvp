# NBA MVP Predictor

Predicts NBA MVP outcomes by learning from **historical MVP voting** (vote share) and producing a **season leaderboard** (2024-25) with explanations.

This repo is structured as a reproducible pipeline:
- **Ingest** historical MVP voting + player season stats
- **Train** a model to predict MVP *vote share* (regression) and evaluate using ranking metrics
- **Predict** a season MVP leaderboard (one row per player); defaults to 2024-25, held out of training
- **Explore** results in a Streamlit app

> If you don't want to scrape Basketball Reference, you can provide your own files in `data/raw/` (schemas below).

---

## Demo

The Streamlit app shows:
- Top-N MVP leaderboard (predicted vote share)
- Player drill-down (inputs + model explanation)
- Filters (min games played, team win %, position)
- Predicted vs. actual voting results, plus backtest performance

---

## Data

All three raw files come from Basketball-Reference (seasons 2009-10 through 2024-25 by default; pass `--start/--end` season ending years to change the range). The scrapers are rate-limited to stay under the site's ~20 requests/minute.

### 1) Historical MVP voting (label)
`data/raw/mvp_voting.csv`: `season` (e.g. `2023-24`), `player`, `vote_share` (float in [0, 1]), `rank`, `is_winner`
```bash
python -m nba_mvp.data.scrape_mvp_voting
```

### 2) Player season stats (features)
`data/raw/player_season_stats.csv`: one row per player-season (traded players use their season totals, with each stint in `teams`). Includes per-game stats (`g`, `mp`, `pts`, `trb`, `ast`, `stl`, `blk`, ...) and advanced stats (`ws`, `ws/48`, `bpm`, `vorp`, `ts%`, `per`, `usg%`, ...).
```bash
python -m nba_mvp.data.scrape_player_season_stats
```

### 3) Team standings (team success feature)
`data/raw/team_standings.csv`: `season`, `team`, `w`, `l`, `win_pct`. Used to give each player a games-weighted team win %.
```bash
python -m nba_mvp.data.scrape_team_standings
```

---

## Quickstart

### 0) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt   # also installs this package in editable mode
```

### 1) Get data (skip if `data/raw/` is already populated)
```bash
python -m nba_mvp.data.scrape_mvp_voting
python -m nba_mvp.data.scrape_player_season_stats
python -m nba_mvp.data.scrape_team_standings
```

### 2) Build training data
```bash
python -m nba_mvp.features.build_training_set
```

### 3) Train + evaluate
The target season is **2024-25**, so the model trains on 2009-10 through 2023-24 only (`--train-through` to change).
```bash
python -m nba_mvp.modeling.train
python -m nba_mvp.modeling.evaluate
```

### 4) Generate the 2024-25 leaderboard
```bash
python -m nba_mvp.modeling.predict_season --season 2024-25
```
Writes `data/processed/leaderboard_2024-25.csv` with predicted vote share, winner probability, heuristic score, and actual voting results for comparison.

### 5) Run the app
```bash
streamlit run app/streamlit_app.py
```

---

## Models

- **Vote share model (primary):** gradient-boosted regressor on season stats, team win %, and within-season percentiles of key stats (MVP voting is relative to that year's league).
- **Winner classifier (secondary):** logistic regression for "won MVP", normalized to sum to 100% within a season.
- **Heuristic score (baseline):** `0.35·PTS + 0.20·AST + 0.15·REB + 0.20·WS + 0.10·BPM`, carried over from the original script.

## Evaluation

This project reports metrics that match the MVP problem:
- **Top-1 accuracy** (did we pick the winner?)
- **Top-3 accuracy**
- **Spearman rank correlation** between predicted and actual vote share (vote-getters only)
- **MAE/RMSE** on vote share (regression)

`evaluate` runs a **walk-forward backtest** (no leakage): each of the last 8 training seasons is predicted by a model trained only on the seasons before it. Results are saved to `reports/backtest.csv` and shown in the app. `train` also prints MAE/RMSE on a two-season holdout.

---

## Repo structure

```txt
.
├── app/                      # Streamlit UI
├── src/nba_mvp/              # package code
│   ├── data/                 # scraping + IO
│   ├── features/             # training table build
│   └── modeling/             # train/eval/predict
├── data/
│   ├── raw/                  # source data (gitignored)
│   └── processed/            # clean tables
├── models/                   # saved model artifacts
└── reports/                  # figures + results
```

---

## Notes / Limitations

MVP voting includes narrative factors (injuries, media, storylines) that may not be fully captured by stats. This model approximates voting patterns from available data and should be interpreted as **a decision-support ranking**, not a guarantee.
