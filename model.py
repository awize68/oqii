# model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Chargement et Préparation des Données ---
df = pd.read_csv('usine_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
# La colonne 'hour' est aussi une bonne feature à ajouter
df['hour'] = df['timestamp'].dt.hour

# Features (X) et Cible (y)
# On utilise maintenant 'day_of_week' qui est numérique
features = ['temperature', 'production_rate', 'hour', 'day_of_week', 'energy_price_omr_per_kwh']
target = 'energy_consumption_kwh'

X = df[features]
y = df[target]

# --- 2. Entraînement du Modèle ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
print("Entraînement du modèle en cours...")
model.fit(X_train, y_train)
print("Entraînement terminé.")

# --- 3. Évaluation du Modèle ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Performance du Modèle ---")
print(f"Erreur Absolue Moyenne (MAE): {mae:.2f} kWh")
print(f"Coefficient de Détermination (R²): {r2:.2f}")

# --- 4. Sauvegarde du Modèle ---
joblib.dump(model, 'energy_predictor.pkl')
print("\nModèle sauvegardé sous le nom 'energy_predictor.pkl'")