import yaml

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import json
import joblib
train_data = pd.read_csv('C:\\Users\\dvid\\Documents\\Python_ML\\Git\\data\\external\\mnist27_train.csv')
test_data = pd.read_csv('C:\\Users\\dvid\\Documents\\Python_ML\\Git\\data\\external\\mnist27_test.csv')

X_train = train_data[['x_1', 'x_2']]
y_train = train_data['y']

X_test = test_data[['x_1', 'x_2']]
y_test = test_data['y']

# Définir le chemin du fichier metrics.json
chemin_fichier = 'metrics.json'

def charger_metrics_json(chemin_fichier):
    """Charger les données existantes du fichier metrics.json."""
    try:
        with open(chemin_fichier, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Si le fichier n'existe pas, retourner un dictionnaire vide
        data = {}
    return data

def sauvegarder_metrics_json(data, chemin_fichier):
    """Sauvegarder les données dans le fichier metrics.json."""
    with open(chemin_fichier, 'w') as f:
        json.dump(data, f, indent=4)


def charger_params_yaml(chemin_fichier):
    """Charger les paramètres à partir du fichier YAML."""
    with open(chemin_fichier, 'r') as fichier:
        params = yaml.safe_load(fichier)
    return params

def entrainer_modele(params, data, chemin_fichier):
    """Entraîner le modèle en fonction des paramètres."""
    modele = params.get('modele', 'modele_par_defaut')

    """Initialisation du dictionnanire data"""
    data = {}
    
    if modele == 'LogisticRegression':
        print("Entraînement du LogisticRegression()...")
        # Code pour entraîner le modèle 1
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

        # Prédiction et évaluation
        y_pred_logistic = logistic_model.predict(X_test)
        accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
        # Ajouter la nouvelle accuracy au dictionnaire existant
        modele_str = str(logistic_model)
        data[modele_str] = accuracy_logistic
        # Sauvegarder les données dans le fichier metrics.json
        sauvegarder_metrics_json(data, chemin_fichier)
        # Sauvegarder le modèle entraîné
        joblib.dump(logistic_model, 'trained_model.pkl')
        print(f"Accuracy Logistic Regression: {accuracy_logistic}")
    elif modele == 'GradientBoostingClassifier':
        # Code pour entraîner le modèle 2
        print("Entraînement du GradientBoostingClassifier()...")
        # Création et entraînement du modèle
        gradient_boosting_model = GradientBoostingClassifier()
        gradient_boosting_model.fit(X_train, y_train)

        # Prédiction et évaluation
        y_pred_gb = gradient_boosting_model.predict(X_test)
        accuracy_gb = accuracy_score(y_test, y_pred_gb)
        # Ajouter la nouvelle accuracy au dictionnaire existant
        modele_str = str(gradient_boosting_model)
        data[modele_str] = accuracy_gb
        # Sauvegarder les données dans le fichier metrics.json
        sauvegarder_metrics_json(data, chemin_fichier)
        # Sauvegarder le modèle entraîné
        joblib.dump(gradient_boosting_model, 'trained_model.pkl')
        print(f"Accuracy Gradient Boosting: {accuracy_gb}")
    else:
        print(f"Modèle {modele} non reconnu. Utilisation du modèle par défaut.")

if __name__ == "__main__":
    # Chemin vers le fichier params.yaml
    chemin_params_yaml = 'C:\\Users\\dvid\\Documents\\Python_ML\\Git\\params.yaml'
    chemin_fichier = 'metrics.json'
    
    # Charger les paramètres à partir du fichier params.yaml
    params = charger_params_yaml(chemin_params_yaml)

     # Charger les données existantes du fichier metrics.json
    data = charger_metrics_json(chemin_fichier)
    
    # Entraîner le modèle en fonction des paramètres
    entrainer_modele(params,data, chemin_fichier)
