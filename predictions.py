import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import ndarray
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model

from utils import *

def load_stats(leagues):
    """
    Load stats from csv files
    :param leagues: list of leagues
    :return: list of dicts containing stats for each league
    """
    result_mapping = {'w': 2, 'd': 1, 'l': 0}
    data = []

    for league in tqdm(leagues):
        for dir in os.listdir("ML-Football-matches-predictor/Data/" + league):
            if dir.startswith("2023"):
                # Get all elements after the first space.
                team = ' '.join(dir.split(' ')[1:])
                
                # Remove the file extension.
                team = '.'.join(team.split('.')[:-1])
                
                df = pd.read_csv(f"ML-Football-matches-predictor/Data/{league}/{dir}")
                df['team'] = team
                df['result'] = df['result'].replace(result_mapping)
                df['momentum'] = rolling_calculation(df, 'result', 5, np.sum)
                df = ratio_calculation(df, 'xG', 'xGA')
                df["xG_xGA_diff"] = df["xG"] - df["xGA"]
                df['avg_xG_last_2'] = rolling_calculation(df, 'xG', 2, np.mean)
                df['avg_xGA_last_2'] = rolling_calculation(df, 'xGA', 2, np.mean)
                df['xG_squared'] = df['xG'] ** 2
                df['xG_cubed'] = df['xG'] ** 3
                df['form_indicator_last_5'] = form_indicator(df, 'result', 5)
                df['win_loss_ratio_last_5'] = win_loss_ratio(df, 'result', 5)
                df['momentum_squared'] = df['momentum'] ** 2
                df['momentum_cubed'] = df['momentum'] ** 3
                df['xG_momentum_interaction'] = df['xG'] * df['momentum']

                # Rolling calculations for last 2 matches
                rolling_columns = ['xG', 'xGA', 'xpts', 'result']
                for column in rolling_columns:
                    df[f'last_two_match_{column}'] = rolling_calculation(df, column, 2, np.sum)
                
                df = interaction_terms(df, 'last_two_match_result', 'xpts')
                df = interaction_terms(df, 'last_two_match_result', 'xG_xGA_ratio')
                df = interaction_terms(df, 'xpts', 'xG_xGA_ratio')
                df['win_ratio_last_5'] = win_ratio_last_5(df, 'result')
                df['weighted_form_last_5'] = weighted_recent_form(df, 'result', 5)
                df['std_dev_result_last_5'] = rolling_calculation(df, 'result', 5, np.std)
                df['net_form_last_5'] = rolling_calculation(df, 'xG', 5, np.sum) - rolling_calculation(df, 'xGA', 5, np.sum)
                df['xG_xpts_interaction'] = df['xG'] * df['xpts']
                df['momentum_xpts_ratio'] = df['momentum'] / df['xpts']
                df.replace([np.inf, -np.inf], 0, inplace=True)  # Replace inf values
                
                columns = ['team', 'momentum', 'last_two_match_xpts', 
                            'last_two_match_result', 'xG_xGA_ratio', 'xG_xGA_diff',
                            'xG_squared', 'form_indicator_last_5', 'win_loss_ratio_last_5', 'momentum_squared', 
                            'momentum_cubed', 'xG_momentum_interaction','last_two_match_result_xpts_interaction', 
                            'last_two_match_result_xG_xGA_ratio_interaction','xpts_xG_xGA_ratio_interaction', 'win_ratio_last_5', 
                            'weighted_form_last_5', 'std_dev_result_last_5','xG_xpts_interaction', 'momentum_xpts_ratio']
                
                df = df[columns]
                
                # garder uniquement la dernier ligne de chaque équipe
                df = df.groupby('team').tail(1)
                
                data.append(df)
                
    data = pd.concat(data)
   
    return data

# Define a function to load the scaler
def load_scaler(path='ML-Football-matches-predictor/scaler.pkl'):
    return joblib.load(path)

# Define a function to standardize features using the saved scaler
def standardize_features(X, scaler_path='ML-Football-matches-predictor/sc.pkl'):
    scaler = load_scaler(scaler_path)
    X_scaled = scaler.transform(X)
    return X_scaled

# Function to load the model and make predictions
def predict_with_model(model_path: str, input_data: pd.DataFrame) -> np.ndarray:
    """
    Load a trained model and make predictions on the input data.
    
    :param model_path: Path to the trained model file (.h5).
    :param input_data: DataFrame containing the input features for prediction.
    :return: Numpy array with the predictions.
    """
    model = load_model(model_path)
    X_scaled = standardize_features(input_data)
    predictions = model.predict(X_scaled)
    predictions = np.round(predictions, 3)
    return pd.DataFrame(predictions, columns=['Win%', 'Draw%', 'Lose%'])

def save_predictions(predictions, path):
    predictions.to_csv(path, index=False)
    
if __name__ == '__main__':
    leagues = ['La_liga', 'Bundesliga', 'EPL', 'Serie_A', 'Ligue_1', 'RFPL'] 
    
    # Load and preprocess the data
    df = load_stats(leagues)
    
    # Drop the 'team' column for prediction purposes
    df_features = df.drop(['team'], axis=1)
    
    # Make predictions
    predictions = predict_with_model('ML-Football-matches-predictor/model.h5', df_features)
    
    # Combine predictions with the team names for better interpretability
    predictions_with_teams = df[['team']].reset_index(drop=True).join(predictions)
    
    print(predictions_with_teams)
    
    today = time.strftime("%d-%m-%Y")
    path = f"ML-Football-matches-predictor/Predictions/Predictons du {today}.csv"
    save_predictions(predictions_with_teams, path)