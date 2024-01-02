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
train_data = pd.read_csv('C:\\Users\\dvid\\Documents\\Python_ML\\Git\\data\\external\\mnist27_train.csv')
test_data = pd.read_csv('C:\\Users\\dvid\\Documents\\Python_ML\\Git\\data\\external\\mnist27_test.csv')

X_train = train_data[['x_1', 'x_2']]
y_train = train_data['y']

X_test = test_data[['x_1', 'x_2']]
y_test = test_data['y']

def charger_params_yaml(chemin_fichier):
    """Charger les paramètres à partir du fichier YAML."""
    with open(chemin_fichier, 'r') as fichier:
        params = yaml.safe_load(fichier)
    return params

def entrainer_modele(params):
    """Entraîner le modèle en fonction des paramètres."""
    modele = params.get('modele', 'modele_par_defaut')
    
    if modele == 'LogisticRegression()':
        print("Entraînement du LogisticRegression()...")
        # Code pour entraîner le modèle 1
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)

        # Prédiction et évaluation
        y_pred_logistic = logistic_model.predict(X_test)
        accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
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
        print(f"Accuracy Gradient Boosting: {accuracy_gb}")
    else:
        print(f"Modèle {modele} non reconnu. Utilisation du modèle par défaut.")

if __name__ == "__main__":
    # Chemin vers le fichier params.yaml
    chemin_params_yaml = 'C:\\Users\\dvid\\Documents\\Python_ML\\Git\\params.yaml'
    
    # Charger les paramètres à partir du fichier params.yaml
    params = charger_params_yaml(chemin_params_yaml)
    
    # Entraîner le modèle en fonction des paramètres
    entrainer_modele(params)
