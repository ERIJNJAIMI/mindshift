from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



app = Flask(__name__)

# Chargement et préparation du modèle
def train_model():
    df = pd.read_csv("C:/Users/fmans/Downloads/Student Mental health.csv")

    # Remplacer les plages de CGPA par des valeurs numériques
    cgpa_mapping = {
        '2.00 - 2.49': 2.25,
        '2.50 - 2.99': 2.75,
        '3.00 - 3.49': 3.25,
        '3.50 - 4.00': 3.75
    }

    df['What is your CGPA?'] = df['What is your CGPA?'].map(cgpa_mapping)
    data_cleaned = df.dropna()

    # Séparation des caractéristiques (X) et de la cible (y)
    X = data_cleaned.drop(columns=["Do you have Depression?"])  # Les caractéristiques
    y = data_cleaned["Do you have Depression?"]  # La cible

    # Encodage des variables catégorielles en numériques
    X = pd.get_dummies(X, drop_first=True)

    # Division des données en ensembles d'entraînement et de test (80% entraînement, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle RandomForestClassifier
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Affichage de l'évaluation du modèle
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

    return model

# Charger le modèle
model = train_model()

# Fonction pour donner des recommandations basées sur la prédiction
def recommend_resources(prediction):
    if prediction == 1:  # Si le modèle prédit que l'utilisateur a de la dépression
        return "Il semble que vous ayez des symptômes de dépression. Nous vous recommandons de consulter un professionnel de la santé mentale. Voici quelques ressources :\n- Article sur la gestion de la dépression\n- Contactez un centre de counseling"
    elif prediction == 0:  # Si le modèle prédit que l'utilisateur n'a pas de dépression
        return "Vous semblez aller bien. Voici quelques ressources pour maintenir votre bien-être mental :\n- Techniques de relaxation\n- Exercices de pleine conscience"
    else:
        return "Nous n'avons pas pu déterminer un état clair. Nous vous conseillons de consulter un professionnel de la santé mentale."

# Route principale
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        course = int(request.form['course'])
        year_of_study = int(request.form['year_of_study'])
        cgpa = float(request.form['cgpa'])
        marital_status = int(request.form['marital_status'])

        # Créer un DataFrame avec les données d'entrée
        input_data = pd.DataFrame({
            'gender': [gender],
            'Age': [age],
            'What is your course?': [course],
            'Your current year of Study': [year_of_study],
            'What is your CGPA?': [cgpa],
            'Marital status': [marital_status]
        })

        # Transformer les données en numériques
        input_data = pd.get_dummies(input_data, drop_first=True)

        # Prédiction du modèle
        prediction = model.predict(input_data)[0]

        # Recommandation basée sur la prédiction
        recommendation = recommend_resources(prediction)

        return render_template('index.html', recommendation=recommendation)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)