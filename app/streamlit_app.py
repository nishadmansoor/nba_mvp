import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="NBA MVP Predictor", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "data" / "processed" / "predictions_current.csv"

st.title("NBA MVP Predictor")
st.caption("Model predicts MVP vote share based on historical voting patterns + available stats.")

if not PRED_PATH.exists():
    st.warning("No predictions file found. Generate it with: `python -m nba_mvp.modeling.predict_current`")
    st.stop()

df = pd.read_csv(PRED_PATH)

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    top_n = st.slider("Top N", 5, 50, 15)
with col2:
    min_games = st.slider("Min games played (G)", 0, 82, 20)
with col3:
    min_win = st.slider("Min team win%", 0.0, 1.0, 0.0)

# Filters
if "g" in df.columns:
    df = df[df["g"].fillna(0) >= min_games]
if "win_pct" in df.columns:
    df = df[df["win_pct"].fillna(0) >= min_win]

df = df.sort_values("pred_vote_share", ascending=False)

st.subheader("Leaderboard")
st.dataframe(df.head(top_n), use_container_width=True)

st.subheader("Player drill-down")
player = st.selectbox("Select a player", df["player"].head(50).tolist() + sorted(set(df["player"]) - set(df["player"].head(50))))
row = df[df["player"] == player].iloc[0]

left, right = st.columns([1,2])
with left:
    st.metric("Predicted vote share", f"{row['pred_vote_share']:.3f}")
    st.write({"team": row.get("team", None)})

with right:
    show_cols = [c for c in df.columns if c not in {"player"}]
    st.write(row[show_cols])

st.markdown("---")
st.caption("Tip: add SHAP/feature importance later for better explanations.")
