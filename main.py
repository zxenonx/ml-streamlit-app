from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from datetime import datetime
import random
import numpy as np
import time
from keras.models import load_model
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import csv

app = FastAPI()

# Charger les modèles entraînés
trained_model_courant_moteur1 = load_model('best_model_courant_moteur1_brut.keras')
trained_model_courant_moteur2 = load_model('best_model_courant_moteur2_brut.keras')
trained_model_pression = load_model('best_model_pression_brut.keras')

# Définir les seuils
SEUIL_DIFFERENCE = 8
SEUIL_DIFFERENCE2 = 0.1
SEUIL_COURANT_MOTEUR1 = 8
SEUIL_COURANT_MOTEUR2 = 8
SEUIL_PRESSION = 8

# Paramètres de connexion SMTP
smtp_server = 'smtp.office365.com'
smtp_port = 587
sender_email = 'yanohterry@outlook.com'
sender_password = 'Fibonacci5813'
receiver_email = 'yanohdemorgan@gmail.com'

# Connexion à la base de données SQLite
def get_db_connection():
    conn = sqlite3.connect('anomalies.db')
    return conn

# Création de la table pour les anomalies
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            timestamp TEXT,
            variable TEXT,
            valeur_predite REAL,
            valeur_lue REAL,
            difference REAL,
            seuil REAL,
            taux_changement REAL,
            rul REAL
        )
    ''')
    conn.commit()
    conn.close()

create_table()

# Variables pour stocker les séquences de données
sequences = {
    "sequence_courant_moteur1": [],
    "sequence_courant_moteur2": [],
    "sequence_charge_moteur1": [],
    "sequence_charge_moteur2": [],
    "sequence_env_temperature": [],
    "sequence_inlet_temperature": [],
    "sequence_outlet_temperature": [],
    "sequence_inlet_pressure": [],
    "sequence_outlet_pressure": [],
    "sequence_humidite": [],
    "sequence_pression": []
}

# Fonction pour calculer la moyenne quadratique d'une séquence de données
def calculate_rms(sequence):
    return np.sqrt(np.mean(np.square(sequence)))

# Fonction pour calculer le taux de changement dans une séquence
def taux_de_changement(sequence):
    differences = [abs((sequence[i + 1] - sequence[i]) / sequence[i]) for i in range(len(sequence) - 1)]
    return np.mean(differences)

# Dénormaliser une valeur prédite ou lue
def denormalize_value(value, min_value, max_value):
    return value * (max_value - min_value) + min_value

# Fonction pour envoyer un e-mail d'alerte
def send_alert(subject, body):
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        print("Mail sent successfully")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'e-mail : {e}")

# Fonction pour enregistrer les anomalies dans la base de données
def record_anomaly(date, timestamp, variable, valeur_predite, valeur_lue, difference, seuil, taux_changement, rul):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO anomalies (date, timestamp, variable, valeur_predite, valeur_lue, difference, seuil, taux_changement, rul)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (date, timestamp, variable, valeur_predite, valeur_lue, difference, seuil, taux_changement, rul))
    conn.commit()
    conn.close()

# Fonction pour générer aléatoirement des données de courants moteurs, température, humidité et pression
def generate_data():
    while True:
        courant_moteur1 = random.uniform(0, 10)  # Générer un courant moteur 1 aléatoire
        courant_moteur2 = random.uniform(0, 10)  # Générer un courant moteur 2 aléatoire
        charge_moteur1 = random.uniform(4, 5)  # Générer une première charge moteur aléatoire
        charge_moteur2 = random.uniform(4, 5)  # Générer une deuxième charge moteur aléatoire
        env_temperature = random.uniform(24, 34)  # Générer une température aléatoire
        inlet_temperature = random.uniform(24, 34)  # Générer une température aléatoire
        outlet_temperature = random.uniform(24, 34)  # Générer une température aléatoire
        inlet_pressure = random.uniform(0, 5)  # Générer une température aléatoire
        outlet_pressure = random.uniform(0, 5)  # Générer une température aléatoire
        humidite = random.uniform(60, 98)  # Générer une humidité aléatoire
        pression = random.uniform(0, 5)  # Générer une pression aléatoire
        timestamp = datetime.now().strftime("%H:%M:%S")  # Obtenir le timestamp actuel

        # Normaliser les données
        normalized_courant_moteur1 = (courant_moteur1 - 0) / (10 - 0)
        normalized_courant_moteur2 = (courant_moteur2 - 0) / (10 - 0)
        normalized_charge_moteur1 = (charge_moteur1 - 4) / (5 - 4)
        normalized_charge_moteur2 = (charge_moteur2 - 4) / (5 - 4)
        normalized_env_temperature = (env_temperature - 24) / (34 - 24)
        normalized_inlet_temperature = (inlet_temperature - 24) / (34 - 24)
        normalized_outlet_temperature = (outlet_temperature - 24) / (34 - 24)
        normalized_inlet_pressure = (inlet_pressure - 0) / (0 - 5)
        normalized_outlet_pressure = (outlet_pressure - 0) / (0 - 5)
        normalized_humidite = (humidite - 60) / (98 - 60)
        normalized_pression = (pression - 0) / (5 - 0)

        yield normalized_courant_moteur1, normalized_courant_moteur2, normalized_charge_moteur1, normalized_charge_moteur2, normalized_env_temperature, normalized_inlet_temperature, normalized_outlet_temperature, normalized_inlet_pressure, normalized_outlet_pressure, normalized_humidite, normalized_pression, timestamp
        time.sleep(1)  # Attendre une seconde avant de générer les prochaines données

# Fonction pour prédire les valeurs RMS et la pression RMS sur les nouvelles données générées
def predict_on_generated_data():
    global sequences

    # Ouvrir un fichier CSV pour écrire les données des RMS et de la pression RMS
    with open('donnees_rms.csv', 'w', newline='') as csvfile:
        fieldnames = ['date', 'timestamp', 'courant_moteur1_rms', 'courant_moteur2_rms', 'charge_moteur1_rms', 'charge_moteur2_rms',
                      'env_temperature_rms', 'inlet_temperature_rms', 'outlet_temperature_rms', 'inlet_pressure_rms', 'outlet_pressure_rms',  'humidite_rms', 'pression_rms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for data_point in generate_data():
            courant_moteur1, courant_moteur2, charge_moteur1, charge_moteur2, env_temperature, inlet_temperature, outlet_temperature, inlet_pressure, outlet_pressure, humidite, pression, timestamp = data_point

            # Ajouter les données à la séquence respective
            sequences["sequence_courant_moteur1"].append(courant_moteur1)
            sequences["sequence_courant_moteur2"].append(courant_moteur2)
            sequences["sequence_charge_moteur1"].append(charge_moteur1)
            sequences["sequence_charge_moteur2"].append(charge_moteur2)
            sequences["sequence_env_temperature"].append(env_temperature)
            sequences["sequence_inlet_temperature"].append(inlet_temperature)
            sequences["sequence_outlet_temperature"].append(outlet_temperature)
            sequences["sequence_inlet_pressure"].append(inlet_pressure)
            sequences["sequence_outlet_pressure"].append(outlet_pressure)
            sequences["sequence_humidite"].append(humidite)
            sequences["sequence_pression"].append(pression)

            # Vérifier si les séquences ont atteint une taille de 10
            if len(sequences["sequence_courant_moteur1"]) == 10:
                # Calculer les valeurs RMS
                rms_values = {key: calculate_rms(value) for key, value in sequences.items()}

                # Écrire les données RMS normalisées dans le fichier CSV avec la date
                writer.writerow({'date': datetime.now().strftime("%Y-%m-%d"),
                                 'timestamp': timestamp,
                                 'courant_moteur1_rms': rms_values["sequence_courant_moteur1"],
                                 'courant_moteur2_rms': rms_values["sequence_courant_moteur2"],
                                 'charge_moteur1_rms': rms_values["sequence_charge_moteur1"],
                                 'charge_moteur2_rms': rms_values["sequence_charge_moteur2"],
                                 'env_temperature_rms': rms_values["sequence_env_temperature"],
                                 'inlet_temperature_rms': rms_values["sequence_inlet_temperature"],
                                 'outlet_temperature_rms': rms_values["sequence_outlet_temperature"],
                                 'inlet_pressure_rms': rms_values["sequence_inlet_pressure"],
                                 'outlet_pressure_rms': rms_values["sequence_outlet_pressure"],
                                 'humidite_rms': rms_values["sequence_humidite"],
                                 'pression_rms': rms_values["sequence_pression"]})

                # Effacer les séquences pour commencer à enregistrer de nouvelles données
                for key in sequences.keys():
                    sequences[key] = sequences[key][5:]

                # Préparer les données d'entrée normalisées pour la prédiction du courant moteur 1
                input_data_courant_moteur1 = np.array(
                    [[rms_values["sequence_courant_moteur1"], rms_values["sequence_charge_moteur1"], rms_values["sequence_env_temperature"], rms_values["sequence_humidite"]]])
                input_data_courant_moteur1 = np.expand_dims(input_data_courant_moteur1, axis=0)  # Ajouter une dimension pour la longueur de la séquence

                # Effectuer la prédiction pour le courant moteur 1
                prediction_courant_moteur1 = trained_model_courant_moteur1.predict(input_data_courant_moteur1)

                prediction_courant_moteur1_denormalized = denormalize_value(prediction_courant_moteur1[0][0], 0, 10)
                rms_courant_moteur1_denormalized = denormalize_value(rms_values["sequence_courant_moteur1"], 0, 10)
                print("Prédiction RMS du courant moteur 1 (dénormalisée) :", prediction_courant_moteur1_denormalized)
                print("Valeur lue du courant moteur 1 (dénormalisée) :", rms_courant_moteur1_denormalized)

                # Calculer la différence entre la valeur prédite et la valeur lue pour le courant moteur 1
                difference_courant_moteur1 = abs(prediction_courant_moteur1_denormalized - rms_courant_moteur1_denormalized)
                print("Différence entre la valeur prédite et la valeur lue pour le courant moteur 1 :", difference_courant_moteur1)

                # Vérifier si la différence pour le courant moteur 1 dépasse le seuil
                if difference_courant_moteur1 > SEUIL_DIFFERENCE:
                    print("Différence entre la valeur prédite et la valeur lue pour le courant moteur 1 dépasse le seuil.")
                    # Calculer le taux de changement pour le courant moteur 1
                    taux_changement_courant_moteur1 = taux_de_changement(sequences["sequence_courant_moteur1"])
                    print("Taux de changement pour le courant moteur 1 :", taux_changement_courant_moteur1)

                    # Calculer le RUL pour le courant moteur 1
                    rul_courant_moteur1 = (SEUIL_COURANT_MOTEUR1 - rms_courant_moteur1_denormalized) / taux_changement_courant_moteur1
                    print("RUL pour le courant moteur 1 :", rul_courant_moteur1)

                    # Enregistrer l'anomalie dans la base de données
                    record_anomaly(datetime.now().strftime("%Y-%m-%d"), timestamp, "courant_moteur1",
                                   prediction_courant_moteur1_denormalized, rms_courant_moteur1_denormalized,
                                   difference_courant_moteur1, SEUIL_DIFFERENCE, taux_changement_courant_moteur1,
                                   rul_courant_moteur1)

                    # Envoi d'un e-mail d'alerte
                    subject = "Alerte : Différence importante dans le courant moteur 1"
                    body = f"La différence entre la valeur prédite ({prediction_courant_moteur1_denormalized}) et la valeur lue ({rms_courant_moteur1_denormalized}) pour le courant moteur 1 dépasse le seuil."
                    send_alert(subject, body)
                # Préparer les données d'entrée normalisées pour la prédiction du courant moteur 2
                input_data_courant_moteur2 = np.array(
                    [[rms_values["sequence_courant_moteur2"], rms_values["sequence_charge_moteur2"], rms_values["sequence_env_temperature"], rms_values["sequence_humidite"]]])
                input_data_courant_moteur2 = np.expand_dims(input_data_courant_moteur2, axis=0)  # Ajouter une dimension pour la longueur de la séquence

                # Effectuer la prédiction pour le courant moteur 2
                prediction_courant_moteur2 = trained_model_courant_moteur2.predict(input_data_courant_moteur2)

                prediction_courant_moteur2_denormalized = denormalize_value(prediction_courant_moteur2[0][0], 0, 10)
                rms_courant_moteur2_denormalized = denormalize_value(rms_values["sequence_courant_moteur2"], 0, 10)
                print("Prédiction RMS du courant moteur 2 (dénormalisée) :", prediction_courant_moteur2_denormalized)
                print("Valeur lue du courant moteur 2 (dénormalisée) :", rms_courant_moteur2_denormalized)

                # Calculer la différence entre la valeur prédite et la valeur lue pour le courant moteur 2
                difference_courant_moteur2 = abs(prediction_courant_moteur2_denormalized - rms_courant_moteur2_denormalized)
                print("Différence entre la valeur prédite et la valeur lue pour le courant moteur 2 :", difference_courant_moteur2)

                # Vérifier si la différence pour le courant moteur 2 dépasse le seuil
                if difference_courant_moteur2 > SEUIL_DIFFERENCE:
                    print("Différence entre la valeur prédite et la valeur lue pour le courant moteur 2 dépasse le seuil.")
                    # Calculer le taux de changement pour le courant moteur 2
                    taux_changement_courant_moteur2 = taux_de_changement(sequences["sequence_courant_moteur2"])
                    print("Taux de changement pour le courant moteur 2 :", taux_changement_courant_moteur2)

                    # Calculer le RUL pour le courant moteur 2
                    rul_courant_moteur2 = (SEUIL_COURANT_MOTEUR2 - rms_courant_moteur2_denormalized) / taux_changement_courant_moteur2
                    print("RUL pour le courant moteur 2 :", rul_courant_moteur2)

                    # Enregistrer l'anomalie dans la base de données
                    record_anomaly(datetime.now().strftime("%Y-%m-%d"), timestamp, "courant_moteur2",
                                   prediction_courant_moteur2_denormalized, rms_courant_moteur2_denormalized,
                                   difference_courant_moteur2, SEUIL_DIFFERENCE, taux_changement_courant_moteur2,
                                   rul_courant_moteur2)

                    # Envoi d'un e-mail d'alerte
                    subject = "Alerte : Différence importante dans le courant moteur 2"
                    body = f"La différence entre la valeur prédite ({prediction_courant_moteur2_denormalized}) et la valeur lue ({rms_courant_moteur2_denormalized}) pour le courant moteur 2 dépasse le seuil."
                    send_alert(subject, body)

                if difference_courant_moteur1 <= SEUIL_DIFFERENCE and difference_courant_moteur2 <= SEUIL_DIFFERENCE:
                    # Préparer les données d'entrée normalisées pour la prédiction de la pression
                    input_data_pression = np.array([[rms_values["sequence_courant_moteur1"], rms_values["sequence_courant_moteur2"], rms_values["sequence_env_temperature"], rms_values["sequence_outlet_pressure"], rms_values["sequence_inlet_pressure"], rms_values["sequence_inlet_temperature"], rms_values["sequence_outlet_temperature"], rms_values["sequence_pression"]]])
                    input_data_pression = np.expand_dims(input_data_pression, axis=0)  # Ajouter une dimension pour la longueur de la séquence

                    # Effectuer la prédiction pour la pression
                    prediction_pression = trained_model_pression.predict(input_data_pression)

                    prediction_pression_denormalized = denormalize_value(prediction_pression[0][0], 0, 5)
                    rms_pression_denormalized = denormalize_value(rms_values["sequence_pression"], 0, 5)
                    print("Prédiction RMS de la pression (dénormalisée) :", prediction_pression_denormalized)
                    print("Valeur lue de la pression (dénormalisée) :", rms_pression_denormalized)

                    # Calculer la différence entre la valeur prédite et la valeur lue pour la pression
                    difference_pression = abs(prediction_pression_denormalized - rms_pression_denormalized)
                    print("Différence entre la valeur prédite et la valeur lue pour la pression :", difference_pression)

                    # Vérifier si la différence pour la pression dépasse le seuil
                    if difference_pression > SEUIL_DIFFERENCE2:
                        print("Différence entre la valeur prédite et la valeur lue pour la pression dépasse le seuil.")
                        # Calculer le taux de changement pour la pression
                        taux_changement_pression = taux_de_changement(sequences["sequence_pression"])
                        print("Taux de changement pour la pression :", taux_changement_pression)

                        # Calculer le RUL pour la pression
                        rul_pression = (SEUIL_PRESSION - rms_pression_denormalized) / taux_changement_pression
                        print("RUL pour la pression :", rul_pression)

                        # Enregistrer l'anomalie dans la base de données
                        record_anomaly(datetime.now().strftime("%Y-%m-%d"), timestamp, "pression",
                                       prediction_pression_denormalized, rms_pression_denormalized,
                                       difference_pression, SEUIL_DIFFERENCE2, taux_changement_pression,
                                       rul_pression)

                        # Envoi d'un e-mail d'alerte
                        subject = "Alerte : Différence importante dans la pression"
                        body = f"La différence entre la valeur prédite ({prediction_pression_denormalized}) et la valeur lue ({rms_pression_denormalized}) pour la pression dépasse le seuil."
                        send_alert(subject, body)

            time.sleep(1)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Monitoring System API"}

@app.get("/start")
def start_predictions(background_tasks: BackgroundTasks):
    background_tasks.add_task(predict_on_generated_data)
    return {"message": "Predictions started"}

@app.get("/anomalies/")
async def get_anomalies():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM anomalies')
    anomalies = cursor.fetchall()
    conn.close()

    result = []
    for anomaly in anomalies:
        result.append({
            "id": anomaly[0],
            "date": anomaly[1],
            "timestamp": anomaly[2],
            "variable": anomaly[3],
            "valeur_predite": anomaly[4],
            "valeur_lue": anomaly[5],
            "difference": anomaly[6],
            "seuil": anomaly[7],
            "taux_changement": anomaly[8],
            "rul": anomaly[9]
        })

    return result

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open('static/index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
