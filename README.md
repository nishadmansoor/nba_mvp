# 🏀 NBA MVP Prediction Model

**📌 Project Overview**

This project analyzes historical MVP data and real-time NBA player performance to predict the NBA MVP for the 2024-2025 season. Using advanced basketball metrics and machine learning, the model provides daily MVP race updates based on the latest player statistics.

**🚀 Features**

- Real-Time MVP Predictions: Fetches live NBA player stats using the NBA API and updates daily.

- Machine Learning Model: Uses Random Forest Regression trained on historical MVP data.

- Custom MVP Score: Combines key basketball metrics (PTS, AST, REB, WS, BPM, PER) to improve prediction accuracy.

- Interactive Streamlit Dashboard: Allows users to compare MVP candidates, track trends, and visualize statistics.

- Automated Data Pipeline: Updates player statistics and MVP leaderboard daily at 3 AM.


**📊 Technologies Used**

- Python (Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn)
- NBA API (for real-time player stats & team standings)
- Machine Learning (Random Forest Regressor, StandardScaler)
- Streamlit (for interactive web dashboard)
- Schedule (for automated data updates)

**🔍 Data & Features**

The model is trained using historical MVP data and enriched with live player stats. Key features include:

| Feature | Description |
| ------------- | ------------- |
| PTS | Points per game |
| AST | Assists per game|
| REB | Rebounds per game|
| WS | Win Shares|
| BPM | Box Plus/Minus|
| PER | Player efficiency rating|
| USG% | Usage|
| TS% | True shooting percentage|
| MVP_Score | Assists per game|

**📈 Model Training & Evaluation**

Data Preprocessing: Handled missing values, and computed rolling averages. 

Feature Engineering: Constructed MVP_Score based on advanced metrics.

Model Used: Random Forest Regressor and Neural Network

Evaluation Metric: Mean Absolute Error (MAE) to assess prediction accuracy for both models. 

**🖥️ How to run the project**

1) Install Necessary Libraries

```python
pip install pandas numpy seaborn matplotlib scikit-learn streamlit schedule nba_api
```

2) Run MVP Prediction & Update Data
```python
python nba_mvp.py
```
3) Launch Streamlit Dashboard
```python
streamlit run nba_mvp.py
```



