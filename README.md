# NBA MVP Predictor

Predicts NBA MVP outcomes by learning from **historical MVP voting** (vote share) and producing a **current-season leaderboard** with explanations.

This repo is structured as a reproducible pipeline:
- **Ingest** historical MVP voting + player season stats
- **Train** a model to predict MVP *vote share* (regression) and evaluate using ranking metrics
- **Predict** a current-season MVP leaderboard (one row per player)
- **Explore** results in a Streamlit app

> If you don't want to scrape Basketball Reference, you can provide your own `data/raw/mvp_voting.csv` and `data/raw/player_season_stats.csv` (schemas below).

---

## Demo

The Streamlit app shows:
- Top-N MVP leaderboard (predicted vote share)
- Player drill-down (inputs + model explanation)
- Filters (min games played, team win%, position/role if available)

---

## Data

### 1) Historical MVP voting (label)
Expected file: `data/raw/mvp_voting.csv`

Minimum columns:
- `season` (e.g., `2023-24`)
- `player`
- `vote_share` (float in [0, 1])
- `rank` (int; optional)
- `team` (optional)

You can generate this by running the scraper:
```bash
python -m nba_mvp.data.scrape_mvp_voting
```

### 2) Player season stats (features)
Expected file: `data/raw/player_season_stats.csv`

Minimum columns:
- `season`, `player`, `team`, `g`, `mp`, `pts`, `trb`, `ast`, `stl`, `blk`
- advanced metrics if available: `ws`, `ws_per_48`, `bpm`, `vorp`, `ts_pct` (any subset is OK)

> For a quick start, this repo includes a cleaned version of your winner-only MVP stats as `data/processed/mvp_winners_stats.csv`. It's useful for sanity checks but **not sufficient** to learn voting patterns by itself.

---

## Quickstart

### 0) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 1) Build training data
```bash
python -m nba_mvp.features.build_training_set
```

### 2) Train + evaluate
```bash
python -m nba_mvp.modeling.train
python -m nba_mvp.modeling.evaluate
```

### 3) Generate current leaderboard
Option A: from an existing file (like your `latest_mvp_predictions.csv`)
```bash
python -m nba_mvp.modeling.predict_current --input data/raw/latest_mvp_predictions.csv
```

### 4) Run the app
```bash
streamlit run app/streamlit_app.py
```

---

## Evaluation

This project reports metrics that match the MVP problem:
- **Top-1 accuracy** (did we pick the winner?)
- **Top-3 accuracy**
- **Spearman rank correlation** between predicted and actual voting ranks
- **MAE/RMSE** on vote share (regression)

The split is **time-based by season** (no leakage):
- Train: older seasons
- Validate/Test: most recent seasons

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
