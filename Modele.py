import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import ndarray
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils import *

def pre_processing(leagues):
    """
    This enhanced function includes additional feature engineering steps to improve the modeling process.
    """
    print('============== pre_processing =================')
    result_mapping = {'w': 2, 'd': 1, 'l': 0}
    data = []
    
    for league in tqdm(leagues):
        for dir in os.listdir("ML-Football-matches-predictor/Data/" + league):
            df = pd.read_csv(f"ML-Football-matches-predictor/Data/{league}/{dir}")
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
            
            df['win_ratio_last_5'] = win_ratio_last_5(df, 'result')
            df['weighted_form_last_5'] = weighted_recent_form(df, 'result', 5)
            df['std_dev_result_last_5'] = rolling_calculation(df, 'result', 5, np.std)
           
            df.replace([np.inf, -np.inf], 0, inplace=True)  # Replace inf values
            data.append(df)

    data = pd.concat(data)
    print("Pre_processing done & the quantity of the data is: ", data.shape[0])
    data = data.dropna()
    features = ['momentum','last_two_match_result','form_indicator_last_2', 'win_loss_ratio_last_2', 'momentum_squared', 
                'momentum_cubed','win_ratio_last_5', 'weighted_form_last_5', 'std_dev_result_last_5',
                'last_match_xG', 'last_match_xGA', 'last_match_xpts']
    
    print("Updated pre_processing done & the quantity of the data is: ", data.shape[0])
    data[features].to_csv("updated_data.csv", index=False)
    print(data[features].head())
    return data[features].values, data['result'].values

def train_model(X: ndarray, y: ndarray) -> Tuple[Sequential, float, float]:
    """
    Build and train a deep learning model for match outcome prediction.
    
    Parameters:
    - X: Features
    - y: Labels
    
    Returns:
    - model: Trained model
    - val_loss: Validation loss
    - val_acc: Validation accuracy
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the deep learning model
    model = Sequential([
        Dense(128, activation='sigmoid', input_shape=(X_train.shape[1],)),  
        Dense(64, activation='sigmoid'),
        Dense(32, activation='sigmoid'),
        Dense(16, activation='sigmoid'),
        Dense(8, activation='sigmoid'),
        Dense(3, activation='softmax')  # 3 classes: win, draw, lose
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    # Train the model with early stopping callback
    history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_val, y_val)
    
    return model, val_loss, val_acc, X_val, y_val, history

if __name__ == "__main__":
    # Define the leagues
    leagues = ['La_liga', 'Bundesliga', 'EPL', 'Serie_A', 'Ligue_1', 'RFPL']    
    
    # Preprocessing the data
    X, y = pre_processing(leagues)
    
    X = standardScaler(X)

    # Train the model
    model, val_loss, val_acc, X_val, y_val, history = train_model(X, y)

    # Print the validation loss and accuracy
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

    # Generate predictions for the validation set
    final_predictions = model.predict(X_val)
    final_predictions = np.argmax(final_predictions, axis=1)

    # Evaluate other metrics
    metrics = evaluate_model(y_val, final_predictions)
    print(f"Evaluation Metrics: {metrics}")
    
    # Save the model
    model.save('ML-Football-matches-predictor/model.h5')

    # Plot confusion matrix
    plot_confusion_matrix(y_val, final_predictions)
    
    # Plot the learning curve
    plot_learning_curve(history)
    
    
    # plot the ROC curve
    plot_multiclass_roc_auc(model, X_val, y_val, 3)
    
    # Define the features
    features = ['momentum','last_two_match_result','form_indicator_last_2', 'win_loss_ratio_last_2', 'momentum_squared', 
                'momentum_cubed','win_ratio_last_5', 'weighted_form_last_5', 'std_dev_result_last_5',
                'last_match_xG', 'last_match_xGA', 'last_match_xpts']    
    # select features importance
    select_featchres_importance(X, y, features)
