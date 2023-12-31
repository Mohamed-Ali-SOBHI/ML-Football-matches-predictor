import time
import pandas as pd
import os
from datetime import datetime, timedelta

def load_matches(target_day):
    """
    Charge les matchs à partir de fichiers CSV.
    :param target_day: jour du match au format 'jj-mm-aaaa'
    :return: liste de DataFrames contenant les matchs pour chaque ligue
    """
    data = []
    base_path = "ML-Football-matches-predictor/MatchOfZeDay"
    
    for file_name in os.listdir(base_path):
        date_str = file_name.split('.')[0].split('_')[-1]
        # Mettre à jour le format ici pour correspondre à 'jj-mm-aa'
        file_date = datetime.strptime(date_str, "%d-%m-%y")

        # Convertit target_day en objet datetime pour la comparaison
        target_date = datetime.strptime(target_day, "%d-%m-%Y")

        # Vérifie si la date du fichier correspond à aujourd'hui, demain ou après-demain
        if file_date in [target_date, target_date + timedelta(days=1), target_date + timedelta(days=2)]:
            print(file_name)
            df = pd.read_csv(f"{base_path}/{file_name}")
            data.append(df)
    
    data = pd.concat(data)
    return data

def load_predictions(day):
    """
    Charge les prédictions à partir de fichiers CSV.
    :return: liste de DataFrames contenant les prédictions pour chaque ligue
    """
    data = []
    base_path = "ML-Football-matches-predictor/Predictions"

    for file_name in os.listdir(base_path):
        date_str = file_name.split('.')[0].split(' ')[-1]
        if date_str == day:
            df = pd.read_csv(f"{base_path}/{file_name}")
            data.append(df)
    
    data = pd.concat(data)
    return data

def resolve_prediction_conflicts(row):
    """
    Identifies conflicts in predictions for Equipe1 and Equipe2.
    If a conflict is identified, a flag is set for later removal of the row.
    :param row: Row of the DataFrame containing predictions for both teams.
    :return: The same row with a flag indicating whether it has conflicting predictions.
    """
    # Set a default value for the conflict flag
    row['conflict'] = False
    
    # If both teams are predicted to win or both are predicted to lose, flag the row
    if (row['Predicted_Result_Equipe1'] == 'Win' and row['Predicted_Result_Equipe2'] == 'Win') or \
       (row['Predicted_Result_Equipe1'] == 'Lose' and row['Predicted_Result_Equipe2'] == 'Lose'):
        row['conflict'] = True
    # If one team is predicted to win and the other to draw, or one to lose and the other to draw, flag the row
    elif (row['Predicted_Result_Equipe1'] == 'Win' and row['Predicted_Result_Equipe2'] == 'Draw') or \
         (row['Predicted_Result_Equipe1'] == 'Draw' and row['Predicted_Result_Equipe2'] == 'Win') or \
         (row['Predicted_Result_Equipe1'] == 'Lose' and row['Predicted_Result_Equipe2'] == 'Draw') or \
         (row['Predicted_Result_Equipe1'] == 'Draw' and row['Predicted_Result_Equipe2'] == 'Lose'):
        row['conflict'] = True
    
    return row


def calculate_expectation_of_gain(row):
    """
    Calculates the expectation of gain for a given row.
    :param row: Row of the DataFrame containing predictions for both teams.
    :return: The same row with the expectation of gain calculated.
    """
    # Extract the odds from the 'cote' column, removing brackets and quotes
    odds = [float(od.strip().replace("'", "")) for od in row['cote'].strip('[]').split(',')]
    
    # Calculate the expectation of gain based on the predicted result
    if row['Predicted_Result_Equipe1'] == 'Win':
        row['expectation of gain'] = row['Confidence_Equipe1'] * odds[0]
    elif row['Predicted_Result_Equipe1'] == 'Draw':
        row['expectation of gain'] = row['Confidence_Equipe1'] * odds[1]
    elif row['Predicted_Result_Equipe1'] == 'Lose':
        row['expectation of gain'] = row['Confidence_Equipe1'] * odds[2]
    
    return row

def merge_data(matches, predictions):
    """
    Fusionne les données de match et de prédiction en appliquant une correspondance de noms d'équipe.
    :param matches: DataFrame contenant les matchs.
    :param predictions: DataFrame contenant les prédictions.
    :return: DataFrame contenant les matchs et les prédictions avec des prédictions pour chaque équipe.
    """
    # Mapping des noms d'équipes connus entre les matchs et les prédictions
    name_mapping = {
        "Paris St Germain": "Paris Saint Germain",
        "Cologne": "FC Cologne",
        "Dortmund": "Borussia Dortmund",
        "Newcastle": "Newcastle United",
        "Sheffield Utd": "Sheffield United",
        "Valence": "Valencia",
        "Clermont": "Clermont Foot",
        "Eintracht Francfort": "Eintracht Frankfurt",
        "Milan": "AC Milan",
        "Heidenheim": "FC Heidenheim",
        "Sociedad": "Real Sociedad",
        "Betis": "Real Betis",
        "Bologne": "Bologna",
        "Bayern": "Bayern Munich",
        "Barcelone": "Barcelona",
        "Gladbach": "Borussia M.Gladbach",
        "Leverkusen": "Bayer Leverkusen",
        "Rb Leipzig": "RasenBallsport Leipzig",
        "Celta": "Celta Vigo",
        "Stuttgart": "VfB Stuttgart",
        "Seville": "Sevilla",
        "Mainz": "Mainz 05",
        "Manchester Utd": "Manchester United",
        "Wolves": "Wolverhampton Wanderers",
        "Athletic Bilbao": "Athletic Club"
    }

    # Apply the mapping to the 'matches' dataframe
    matches['Equipe1'] = matches['Equipe1'].str.title().replace(name_mapping)
    matches['Equipe2'] = matches['Equipe2'].str.title().replace(name_mapping)

    # Merge predictions for Equipe1
    merged_with_equipe1 = pd.merge(
        matches,
        predictions,
        left_on='Equipe1',
        right_on='team',
        how='left'
    )
    merged_with_equipe1.rename(columns={'Predicted_Result': 'Predicted_Result_Equipe1', 'Confidence': 'Confidence_Equipe1'}, inplace=True)

    # Merge predictions for Equipe2
    merged_with_equipe2 = pd.merge(
        merged_with_equipe1,
        predictions,
        left_on='Equipe2',
        right_on='team',
        how='left',
        suffixes=('', '_Equipe2')
    )
    merged_with_equipe2.rename(columns={'Predicted_Result': 'Predicted_Result_Equipe2', 'Confidence': 'Confidence_Equipe2'}, inplace=True)

    # Drop the extra 'team' columns
    final_merged = merged_with_equipe2.drop(columns=['team', 'team_Equipe2'])
    
    filtred_final_merged = final_merged.apply(resolve_prediction_conflicts, axis=1)
    filtred_final_merged = filtred_final_merged[filtred_final_merged['conflict'] == False].drop(columns='conflict')
    
    filtred_final_merged = filtred_final_merged.apply(calculate_expectation_of_gain, axis=1)
    
    return filtred_final_merged

def save_te9mira(filtred_final_merged, day):
    filtred_final_merged.to_csv(f"ML-Football-matches-predictor/9amir/ta9mira du {day}.csv", index=False)
    
current_day = time.strftime("%d-%m-%Y")
matches = load_matches(current_day)

predictions = load_predictions(current_day)

filtred_merge_df = merge_data(matches, predictions) 
print(filtred_merge_df)

save_te9mira(filtred_merge_df, current_day)