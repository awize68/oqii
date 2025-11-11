# data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Paramètres de simulation ---
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 3, 1)
num_hours = int((end_date - start_date).total_seconds() / 3600)

# --- Génération des données ---
timestamps = [start_date + timedelta(hours=i) for i in range(num_hours)]
data = []

for ts in timestamps:
    hour = ts.hour
    day_of_week = ts.weekday()  # Lundi=0, Mardi=1, ..., Dimanche=6
    
    # Simuler la température (plus chaude le jour, pics aléatoires)
    base_temp = 25 + 10 * np.sin((hour - 6) * np.pi / 12)
    heat_wave = 5 if np.random.rand() > 0.98 else 0
    temperature = base_temp + heat_wave + np.random.normal(0, 1.5)
    
    # Simuler la production (plus élevée en semaine)
    base_production = 80 if day_of_week < 5 else 20  # Si jour < 5 (Lundi à Vendredi)
    production_rate = base_production + np.random.normal(0, 5)
    production_rate = max(0, production_rate)
    
    # Simuler la consommation énergétique
    energy_consumption = (0.8 * production_rate) + (1.5 * temperature) + np.random.normal(0, 5)
    energy_consumption = max(10, energy_consumption)

    # Simuler le coût de l'énergie (heures pleines/creuses)
    if 8 <= hour <= 11 or 17 <= hour <= 21:
        energy_price = 0.15 # Heures pleines
    else:
        energy_price = 0.08 # Heures creuses
        
    # --- MODIFICATION CLÉ ---
    # On utilise day_of_week (qui est un nombre) directement
    data.append([ts, temperature, production_rate, energy_consumption, energy_price, day_of_week])

df = pd.DataFrame(data, columns=['timestamp', 'temperature', 'production_rate', 'energy_consumption_kwh', 'energy_price_omr_per_kwh', 'day_of_week'])
df.to_csv('usine_data.csv', index=False)

print("Fichier 'usine_data.csv' généré avec succès !")
print(df.head())