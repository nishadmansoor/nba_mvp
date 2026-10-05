import joblib
import pandas as pd
import streamlit as st

from nba_mvp.data.io import PROCESSED_DIR
from nba_mvp.modeling.evaluate import BACKTEST_PATH
from nba_mvp.modeling.predict_season import explain, rotation_baseline
from nba_mvp.modeling.train import MODEL_PATH

FEATURE_LABELS = {
    "g": "Games", "mp": "Minutes/game", "pts": "Points/game", "trb": "Rebounds/game",
    "ast": "Assists/game", "stl": "Steals/game", "blk": "Blocks/game",
    "ws": "Win Shares", "ws_per_48": "WS/48", "bpm": "BPM", "vorp": "VORP",
    "ts_pct": "True shooting %", "per": "PER", "usg_pct": "Usage %", "win_pct": "Team win %",
}
for _c in ["pts", "ast", "trb", "ws", "bpm", "vorp", "per", "win_pct"]:
    FEATURE_LABELS[f"{_c}_season_pct"] = f"{FEATURE_LABELS[_c]} (pctile)"

# -------------------------------
# App configuration
# -------------------------------
st.set_page_config(page_title="NBA MVP Tracker", layout="wide")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_leaderboard(path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

leaderboards = {p.stem.removeprefix("leaderboard_"): p for p in sorted(PROCESSED_DIR.glob("leaderboard_*.csv"))}
if not leaderboards:
    st.error("Leaderboard data not found. Run `python -m nba_mvp.modeling.predict_season` first.")
    st.stop()

models = load_models()

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.title("Controls")

season = st.sidebar.selectbox("Season", list(leaderboards)[::-1])
df = load_leaderboard(leaderboards[season])
has_actual = "actual_vote_share" in df.columns

metric_map = {
    "Predicted Vote Share": "pred_vote_share",
    "Winner Probability": "pred_winner_prob",
    "Heuristic Score": "heuristic_score",
}
if has_actual:
    metric_map["Actual Vote Share"] = "actual_vote_share"

selected_metric_label = st.sidebar.radio("Ranking metric", list(metric_map.keys()), index=0)
rank_col = metric_map[selected_metric_label]

min_games = st.sidebar.slider("Minimum games played", 0, int(df["g"].max()), 50)
min_win_pct = st.sidebar.slider("Minimum team win %", 0.0, 1.0, 0.0, step=0.05)
positions = sorted(df["pos"].dropna().unique())
selected_pos = st.sidebar.multiselect("Positions", positions, default=positions)
top_n = st.sidebar.selectbox("Show Top N players", [5, 10, 15, 20], index=1)

# -------------------------------
# Filter + rank
# -------------------------------
df_filtered = df[
    (df["g"] >= min_games)
    & (df["win_pct"].fillna(0) >= min_win_pct)
    & (df["pos"].isin(selected_pos))
].copy()
df_filtered = df_filtered.sort_values(rank_col, ascending=False)
df_filtered["rank"] = range(1, len(df_filtered) + 1)

# -------------------------------
# Header
# -------------------------------
st.title(f"🏀 NBA MVP Tracker — {season} Season")
caption = "MVP leaderboard based on historical voting patterns and player performance."
if models is not None and season > models["train_through"]:
    caption += f" Model trained on {models['train_seasons'][0]} – {models['train_through']} only, so {season} is out-of-sample."
st.caption(caption)

if df_filtered.empty:
    st.warning("No players match these filters.")
    st.stop()

# -------------------------------
# Top metrics
# -------------------------------
top_player = df_filtered.iloc[0]

cols = st.columns([3, 1, 2, 2, 3] if has_actual else [3, 1, 2, 2])
cols[0].metric("Current #1", top_player["player"])
cols[1].metric("Team", top_player["team"])
cols[2].metric("Vote Share (pred.)", f"{top_player['pred_vote_share']:.3f}")
cols[3].metric("Winner Prob.", f"{top_player['pred_winner_prob']:.1%}")
if has_actual:
    actual_winner = df.sort_values("actual_vote_share", ascending=False).iloc[0]
    cols[4].metric("Actual MVP", actual_winner["player"], f"{actual_winner['actual_vote_share']:.3f} share",
                   delta_color="off")

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Leaderboard", "Player Deep Dive", "Model Performance", "Methodology"])

# -------------------------------
# Leaderboard tab
# -------------------------------
with tab1:
    st.subheader(f"Top {top_n} MVP Candidates")

    display_cols = [
        "rank", "player", "team", "pos", "g",
        "pts", "trb", "ast", "ws", "bpm",
        "win_pct",
        "pred_vote_share", "pred_winner_prob", "heuristic_score",
    ]
    if has_actual:
        display_cols += ["actual_vote_share", "actual_rank"]

    st.dataframe(
        df_filtered[display_cols].head(top_n),
        hide_index=True,
        width="stretch",
        column_config={
            "pos": "Pos", "g": "G", "pts": "PTS", "trb": "REB", "ast": "AST", "ws": "WS", "bpm": "BPM",
            "win_pct": st.column_config.NumberColumn("Team Win %", format="%.3f"),
            "pred_vote_share": st.column_config.NumberColumn("Vote Share (pred.)", format="%.3f"),
            "pred_winner_prob": st.column_config.NumberColumn("Winner Prob.", format="percent"),
            "heuristic_score": st.column_config.NumberColumn("Heuristic", format="%.2f"),
            "actual_vote_share": st.column_config.NumberColumn("Vote Share (actual)", format="%.3f"),
            "actual_rank": st.column_config.NumberColumn("Actual Rank", format="%d"),
        },
    )

# -------------------------------
# Player deep dive
# -------------------------------
with tab2:
    st.subheader("Player Deep Dive")

    player_name = st.selectbox("Select a player", df_filtered["player"].unique())
    player_row = df_filtered[df_filtered["player"] == player_name].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PTS", f"{player_row['pts']:.1f}")
    c2.metric("REB", f"{player_row['trb']:.1f}")
    c3.metric("AST", f"{player_row['ast']:.1f}")
    c4.metric("Win Shares", f"{player_row['ws']:.1f}")
    c5.metric("Team Win %", f"{player_row['win_pct']:.3f}")

    st.markdown("### Model Scores")
    s = st.columns(4 if has_actual else 3)
    s[0].metric("Predicted Vote Share", f"{player_row['pred_vote_share']:.3f}")
    s[1].metric("Winner Probability", f"{player_row['pred_winner_prob']:.1%}")
    s[2].metric("Heuristic Score", f"{player_row['heuristic_score']:.2f}")
    if has_actual:
        s[3].metric("Actual Vote Share", f"{player_row['actual_vote_share']:.3f}")

    st.markdown("### What's driving the prediction")
    if models is None:
        st.info("Train the model to see explanations.")
    else:
        feature_cols = models["feature_cols"]
        row = df[df["player"] == player_name].head(1)
        contribs = explain(models, row, rotation_baseline(df, feature_cols)).head(8)
        st.caption(
            "Change in predicted vote share if this stat were replaced with a typical rotation "
            "player's value (median of players averaging 20+ minutes). Positive = helps the MVP case."
        )
        chart = pd.DataFrame({
            "stat": [FEATURE_LABELS.get(c, c) for c in contribs.index],
            "impact": contribs.values,
        })
        st.bar_chart(chart, x="stat", y="impact", horizontal=True, sort="-impact")

# -------------------------------
# Model performance
# -------------------------------
with tab3:
    st.subheader("Walk-forward backtest")
    if not BACKTEST_PATH.exists():
        st.info("Run `python -m nba_mvp.modeling.evaluate` to generate backtest results.")
    else:
        bt = pd.read_csv(BACKTEST_PATH)
        st.caption("Each season is predicted by a model trained only on the seasons before it.")
        m = st.columns(4)
        m[0].metric("Top-1 accuracy", f"{bt['top1'].mean():.0%}")
        m[1].metric("Top-3 accuracy", f"{bt['top3'].mean():.0%}")
        m[2].metric("Spearman (vote-getters)", f"{bt['spearman_vote_getters'].mean():.2f}")
        m[3].metric("Heuristic top-1", f"{bt['heuristic_top1'].mean():.0%}")
        st.dataframe(
            bt[["season", "actual_winner", "predicted_winner", "top1", "top3", "classifier_top1",
                "spearman_vote_getters", "mae"]],
            hide_index=True, width="stretch",
        )

# -------------------------------
# Methodology
# -------------------------------
with tab4:
    st.markdown(
        f"""
### Methodology

This dashboard estimates NBA MVP outcomes for the **{season} season** using three complementary approaches:

**1. Vote Share Model (Primary)**
A gradient-boosted regression model trained on historical MVP voting data from Basketball Reference,
using season-level box score and advanced stats, team win %, and each stat's percentile within its season.

**2. Winner Classifier (Secondary)**
A logistic regression trained to identify each season's MVP winner. Probabilities are normalized so they
sum to 100% across a season, and serve as a sanity check on ranking quality.

**3. Heuristic Score (Baseline)**
A custom MVP score inspired by common media narratives
(0.35·PTS + 0.20·AST + 0.15·REB + 0.20·WS + 0.10·BPM).
This score is used as a baseline and interpretability tool, not as ground truth.

MVP voting includes narrative factors (injuries, media, storylines) that may not be fully captured by stats,
so treat the rankings as decision support rather than a guarantee.
"""
    )
