from itertools import cycle
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def rolling_calculation(df, column, window, func):
    return df[column].rolling(window, min_periods=1).apply(func)

def replace_inf(df, column):
    df[column].replace([np.inf, -np.inf], 0, inplace=True)
    return df

def ratio_calculation(df, column1, column2):
    df[f"{column1}_{column2}_ratio"] = df[column1] / df[column2]
    df = replace_inf(df, f"{column1}_{column2}_ratio")
    return df
    
def form_indicator(df, column, window):
    return df[column].rolling(window, min_periods=1).mean()

def win_loss_ratio(df, column, window):
    return df[column].rolling(window, min_periods=1).apply(lambda x: np.sum(x == 2) / np.sum(x == 0) if np.sum(x == 0) != 0 else 0)
    
def interaction_terms(df, col1, col2):
    df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
    return df

def win_ratio_last_5(df, column):
    return df[column].rolling(window=5, min_periods=1).apply(lambda x: np.sum(x == 2) / 5)
    
def weighted_recent_form(df, column, window):
    weights = np.arange(1, window + 1)
    return df[column].rolling(window, min_periods=1).apply(lambda x: np.dot(x[-window:], weights[-len(x):]) / np.sum(weights[-len(x):]))

def last_match_metric(df, column):
    return df[column].shift(1)

def standardScaler(X):
    """
    This function is used to standardize the data
    :param X: the features
    :return: X
    """
    sc = StandardScaler()
    X = sc.fit_transform(X)
    # save the standard scaler
    joblib.dump(sc, 'ML-Football-matches-predictor/sc.pkl')
    return X

def pca(X):
    """
    This function is used to reduce the dimension of the data
    :param X: the features
    :return: X
    """
    pca = PCA(n_components=10)
    X = pca.fit_transform(X)
    # save the pca
    joblib.dump(pca, 'ML-Football-matches-predictor/pca.pkl')
    return X

def select_featchres_importance(X, y, features):
    """
    This function is used to select the features
    :param X: the features
    :param y: the target
    :return: X
    """
    # Train a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf.fit(X, y)  # Utilisez les données que vous avez déjà préparées
    importances = rf.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
    print(feature_importance_df)

    # Plotting the feature importances
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Evaluate the model using various metrics
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }
    return metrics

def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    """
    Plot confusion matrix using seaborn with percentages
    :param y_true: True labels
    :param y_pred: Predicted labels
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize the confusion matrix
    cm_percentage = cm_normalized * 100  # convert to percentages
    
    labels = ['lose', 'draw', 'win']  # define the labels based on your mapping

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (in %)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def cross_validation(X, y, build_fn, cv=5):
    """
    Perform cross validation.

    Args:
        X (array-like): The features.
        y (array-like): The target.
        build_fn (callable): Function to build the Keras model.
        cv (int, optional): Number of folds. Defaults to 5.

    Returns:
        numpy.ndarray: Array of scores of the estimator for each run of the cross validation.
    """
    model = KerasClassifier(build_fn=build_fn)
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean()

def plot_learning_curve(history):
    """
    Trace la courbe d'apprentissage du modèle en fonction de l'historique d'entraînement.

    Parameters:
    - history: Historique de retour de la fonction fit de Keras, contenant les métriques d'entraînement.
    """
    # Récupération des données de précision et de perte
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Tracé de la précision
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Tracé de la perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_multiclass_roc_auc(model, X, y, n_classes):
    """
    Trace les courbes ROC et calcule l'AUC pour un modèle de classification multiclasse.

    Parameters:
    - model: Modèle entraîné.
    - X: Données de test/validation.
    - y: Étiquettes réelles.
    - n_classes: Nombre de classes.
    """
    # Binarisation des étiquettes
    y_bin = label_binarize(y, classes=range(n_classes))

    # Calcul des scores de probabilité
    y_score = model.predict(X)

    # Structures pour le tracé
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calcul des courbes ROC et AUC pour chaque classe
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Configuration du tracé
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])
    plt.figure(figsize=(7, 7))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.show()

def save_model(model):
    """
    This function is used to save the model
    :param model: the model
    :return: None
    """
    model.save('model.h5')
    print("Model saved successfully")
    