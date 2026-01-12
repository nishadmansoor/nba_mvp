import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import schedule
import time
import threading
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
from datetime import datetime
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.endpoints import leaguedashteamstats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# load & clean data
def load_data():
    try:
        file_path = 'all_mvp_stats.csv'
        mvp_data = pd.read_csv(file_path, skiprows=1)
        print("Data loaded successfully")
        print(mvp_data.head())

        # mapping actual column names to expected ones
        column_mapping = {
            "Lg": "League",
            "Tm": "Team",
            "G": "Games_Played",
            "MP": "MPG",
            "WS/48": "WS_per_48"
        }

        # renaming columns
        mvp_data.rename(columns=column_mapping, inplace=True)

        # Ensure missing columns exist
        required_columns = ["Season", "League", "Player", "PTS", "TRB", "AST", "STL", "BLK", "FG%", "3P%", "FT%", "WS", "WS_per_48", "BPM", "PER", "USG%", "TS%", "VORP"]
        for col in required_columns:
            if col not in mvp_data.columns:
                mvp_data[col] = np.nan 

        # Convert numeric columns
        numeric_cols = ["PTS", "TRB", "AST", "STL", "BLK", "FG%", "3P%", "FT%", "WS", "WS_per_48", "BPM", "PER", "USG%", "TS%", "VORP"]
        existing_numeric_cols = [col for col in numeric_cols if col in mvp_data.columns]
        mvp_data[existing_numeric_cols] = mvp_data[existing_numeric_cols].apply(pd.to_numeric, errors='coerce')


        mvp_data["MVP_Score"] = (
            mvp_data["PTS"] * 0.35 +
            mvp_data["AST"] * 0.20 +
            mvp_data["TRB"] * 0.15 + 
            mvp_data["WS"] * 0.2 +
            mvp_data["BPM"] * 0.1
        )

        #print("MVP data loaded successfully")
        return mvp_data

    except Exception as e:
        print(f"Error loading all_mvp_stats.csv: {e}")
        return pd.DataFrame()

# get game info
def get_logs():
    game_logs = playergamelogs.PlayerGameLogs(season_nullable = '2024-25').get_data_frames()[0]
    game_logs = game_logs[['PLAYER_NAME', 'GAME_DATE', 'TEAM_NAME', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN']]
    game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
    return game_logs

# process stats
def process_stats(game_logs):
    game_logs['PTS_10G'] = game_logs.groupby('PLAYER_NAME')['PTS'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    game_logs['REB_10G'] = game_logs.groupby('PLAYER_NAME')['REB'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    game_logs['AST_10G'] = game_logs.groupby('PLAYER_NAME')['AST'].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
    latest_stats = game_logs.sort_values('GAME_DATE').groupby('PLAYER_NAME').last().reset_index()
    return latest_stats

# team standings
def team_standings():
    team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2024-25').get_data_frames()[0]
    team_stats = team_stats[['TEAM_NAME', 'W', 'L', 'GP']]
    team_stats['WIN_PCT'] = team_stats['W'] / team_stats['GP']
    return team_stats

# merge data
def merge_data(player_stats, team_stats):
    player_stats = player_stats.merge(team_stats, on='TEAM_NAME', how='left')
    player_stats['WIN_PCT'] = player_stats['WIN_PCT'].fillna(0)

    player_stats['MVP_Score'] = (
        player_stats['PTS_10G'] * 0.3 +
        player_stats['REB_10G'] * 0.3 +
        player_stats['AST_10G'] * 0.3 +
        player_stats['WIN_PCT'] * 0.1
    )
    return player_stats

# eliminating the injured players from the model
def injured_players():
    url = "https://www.basketball-reference.com/friv/injuries.fcgi"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error in fetching data: ", response.status_code)
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    injury_table = soup.find("table", {"id": "injuries"})

    if not injury_table: 
        print("Could not find the injury table")
        return []
    
    injured_players = []

    for row in injury_table.find_all("tr")[1:]:
        columns = row.find_all("td")
        if len(columns) > 1:
            player_name = columns[0].text.strip()
            status = columns[2].text.strip()

            if "Out" in status or "Season" in status: 
                injured_players.append(player_name)
    return injured_players


# train predictive model
def train_model(player_stats):
    previous_mvp = load_data()

    if previous_mvp.empty:
        return None, None, []

    data = player_stats.merge(previous_mvp, left_on='PLAYER_NAME', right_on='Player', how='left').fillna(0)

    if 'MVP_Score_x' in data.columns:
        data.rename(columns={'MVP_Score_x': 'MVP_Score'}, inplace=True)
    elif 'MVP_Score_y' in data.columns:
        data.rename(columns={'MVP_Score_y': 'MVP_Score'}, inplace=True)

    feature_columns = ['PTS_10G', 'REB_10G', 'AST_10G', 'WIN_PCT']
    X = data[feature_columns]
    y = data['MVP_Score']

    # Random Forest Regressor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Random Forest MAE: {mae:.4f}")

    # Pytorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    Y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    class MVPNet(nn.Module):
        def __init__(self, input_size):
            super(MVPNet, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(X_train_tensor.shape[1], 16),
                nn.ReLU(),
                nn.Linear(16,1)
            )
        def forward(self, x):
            return self.net(x)
        
    model_nn = MVPNet()
    loss_nn = nn.MSELoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.01)

    for epoch in range(300):
        model_nn.train()
        optimizer.zero_grad()
        output = model_nn(X_train_tensor)
        loss = loss_nn(output, Y_train_tensor)
        loss.backward()
        optimizer.step()

    model_nn.eval()
    with torch.no_grad():
        prediction = model_nn(X_test_tensor)
        mae_nn = mean_absolute_error(Y_test_tensor.numpy(), prediction.numpy())
        print(f"Pytorch MAE: {mae_nn:.4f}")
    return model, scaler, feature_columns

# update the predictions
final_game = datetime(2025, 4, 13)
def update_predictions():
    if datetime.now() > final_game:
        print("The NBA season has ended. No further updates")
        return
    
    game_logs = get_logs()
    player_stats = process_stats(game_logs)
    team_stats = team_standings()
    merged_data = merge_data(player_stats, team_stats)

    model, scaler, _ = train_model(merged_data)

    merged_data.to_csv('latest_mvp_predictions.csv', index=False)
    #print("Updated predictions")    

# streamlit dashboard
def dashboard():
    st.title("NBA MVP Race - Live Predictions")
    latest_mvp = pd.read_csv('latest_mvp_predictions.csv')
    st.write("Loaded latest MVP data succesfully!")

    if 'MVP_Score' not in latest_mvp.columns:
        raise KeyError("Column 'MVP_Score' not found. Check available columns above")
    
    out_for_season = injured_players()
    latest_mvp = latest_mvp[~latest_mvp['PLAYER_NAME'].isin(out_for_season)]

    st.subheader("Top 10 in the MVP Race")
    top_10_mvp = latest_mvp[['PLAYER_NAME', 'MVP_Score']].sort_values('MVP_Score', ascending=False).head(10)

    top_10_mvp.reset_index(drop=True, inplace=True)
    top_10_mvp.insert(0, "Rank", range(1, len(top_10_mvp) + 1))

    st.dataframe(top_10_mvp)

    player_name = st.selectbox("Choose a player", latest_mvp['PLAYER_NAME'])
    if st.button("Show MVP Prediction"):
       # player_data = latest_mvp[latest_mvp['PLAYER_NAME'] == player_name].drop(columns=['PLAYER_NAME'])
        model, scaler, feature_columns = train_model(latest_mvp)
        if model is None or scaler is None:
            st.write("Model not trained. Please try again later.")
        else:
            player_data = latest_mvp[latest_mvp['PLAYER_NAME'] == player_name][feature_columns]
            X_scaled = scaler.transform(player_data)
            prediction = model.predict(X_scaled)[0]

            st.write(f"Predicted MVP Score: {prediction * 100:.2f}")
            st.write("Chance to win MVP: ", "Yes" if prediction > 0.5 else "No")

if __name__ == "__main__":
    update_predictions()

    schedule.every().day.at("03:00").do(update_predictions)

    dashboard()

    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    threading.Thread(target=run_scheduler, daemon=True).start()