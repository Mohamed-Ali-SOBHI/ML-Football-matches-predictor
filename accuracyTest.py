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
        for dir in os.listdir("ML-Football-matches-predictor/StatsOfZisYear/" + league):
            if dir.startswith("2023"):
                # Get all elements after the first space.
                team = ' '.join(dir.split(' ')[1:])
                
                # Remove the file extension.
                team = '.'.join(team.split('.')[:-1])
                
                df = pd.read_csv(f"ML-Football-matches-predictor/StatsOfZisYear/{league}/{dir}")
                df = df.append(pd.Series(), ignore_index=True)
                df['team'] = team
                df['result'] = df['result'].replace(result_mapping)
                
                df['momentum'] = rolling_calculation(df, 'result', 2, np.sum)
                df['momentum_squared'] = df['momentum'] ** 2
                df['momentum_cubed'] = df['momentum'] ** 3
                
                df['last_match_xG'] = last_match_metric(df, 'xG')
                df['last_match_xGA'] = last_match_metric(df, 'xGA')
                df['last_match_xpts'] = last_match_metric(df, 'xpts')
                
                df['avg_xG_last_2'] = rolling_calculation(df, 'xG', 2, np.mean)
                df['avg_xGA_last_2'] = rolling_calculation(df, 'xGA', 2, np.mean)
                df['form_indicator_last_2'] = form_indicator(df, 'result', 2)
                df['win_loss_ratio_last_2'] = win_loss_ratio(df, 'result', 2)            

                # Rolling calculations for last 2 matches
                rolling_columns = ['xG', 'xGA', 'xpts', 'result']
                for column in rolling_columns:
                    df[f'last_two_match_{column}'] = rolling_calculation(df, column, 2, np.sum)
                
                df = interaction_terms(df, 'last_two_match_result', 'xpts')
                df['win_ratio_last_5'] = win_ratio_last_5(df, 'result')
                df['weighted_form_last_5'] = weighted_recent_form(df, 'result', 5)
                df['std_dev_result_last_5'] = rolling_calculation(df, 'result', 5, np.std)
                
                df.replace([np.inf, -np.inf], 0, inplace=True)  # Replace inf values
                
                features = ['team', 'momentum','last_two_match_result','form_indicator_last_2', 'win_loss_ratio_last_2', 'momentum_squared', 
                            'momentum_cubed','win_ratio_last_5', 'weighted_form_last_5', 'std_dev_result_last_5',
                            'last_match_xG', 'last_match_xGA', 'last_match_xpts', 'result']
                
                df = df[features]
                
                data.append(df)
                
    data = pd.concat(data)
    
    data.to_csv("ML-Football-matches-predictor/StatsOfZisYear/2023.csv", index=False)
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
    predictions = np.round(predictions, 2)
    return pd.DataFrame(predictions, columns=['Win%', 'Draw%', 'Lose%'])
    
if __name__ == '__main__':
    leagues = ['La_liga', 'Bundesliga', 'EPL', 'Serie_A', 'Ligue_1']
    
    # Load and preprocess the data
    df = load_stats(leagues)
    
    # Drop the 'team' column for prediction purposes
    df_features = df.drop(['team', 'result'], axis=1)
    
    # Make predictions
    predictions = predict_with_model('ML-Football-matches-predictor/model.h5', df_features)
    
    # Combine predictions with the team names for better interpretability
    predictions_with_teams = df[['team', 'result']].reset_index(drop=True).join(predictions)
    
    predictions_with_teams['Predicted_Result'] = predictions_with_teams[['Win%', 'Draw%', 'Lose%']].idxmax(axis=1)
    predictions_with_teams = predictions_with_teams.dropna()
    
    predictions_with_teams['Predicted_Result'] = predictions_with_teams['Predicted_Result'].replace({'Win%': 'Lose%', 'Lose%': 'Win%'})
    
    # Créer un mapping des noms de colonnes aux valeurs de résultat
    result_mapping = {'Win%': 2, 'Draw%': 1, 'Lose%': 0}

    # Mapper les prédictions aux valeurs de résultat
    predictions_with_teams['Predicted_Value'] = predictions_with_teams['Predicted_Result'].map(result_mapping)
    predictions_with_teams.to_csv("ML-Football-matches-predictor/predictions.csv", index=False)
    
    # Calculer la précision
    accuracy = (predictions_with_teams['Predicted_Value'] == predictions_with_teams['result']).mean()
    print(f"Accuracy: {accuracy}")
