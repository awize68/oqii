# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Configuration de la page ---
st.set_page_config(
    page_title="Cortex Énergétique - POC OQII",
    page_icon="⚡",
    layout="wide"
)

# --- Fonctions utilitaires ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('usine_data.csv')
        st.success("Fichier de données 'usine_data.csv' chargé avec succès.")
        return df
    except FileNotFoundError:
        st.error("ERREUR : Le fichier 'usine_data.csv' est introuvable. Assurez-vous d'avoir lancé data_generator.py d'abord.")
        st.stop() # Arrête l'exécution du script

@st.cache_resource
def load_model():
    try:
        model = joblib.load('energy_predictor.pkl')
        st.success("Modèle 'energy_predictor.pkl' chargé avec succès.")
        return model
    except FileNotFoundError:
        st.error("ERREUR : Le fichier 'energy_predictor.pkl' est introuvable. Assurez-vous d'avoir lancé model.py d'abord.")
        st.stop() # Arrête l'exécution du script

def predict_consumption(model, input_df):
    return model.predict(input_df)

# --- Chargement des ressources ---
df = load_data()
model = load_model()
df['timestamp'] = pd.to_datetime(df['timestamp'])


# --- Interface du Dashboard ---
st.title("⚡ Cortex Énergétique - Tableau de Bord Opérationnel")
st.markdown("POC pour OQ for Industrial Investments - Optimisation des Coûts Énergétiques")

# --- KPIs en Haut ---
st.header("📊 Indicateurs Clés de Performance (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_consumption = df['energy_consumption_kwh'].sum()
total_cost = (df['energy_consumption_kwh'] * df['energy_price_omr_per_kwh']).sum()
avg_temp = df['temperature'].mean()
avg_production = df['production_rate'].mean()

col1.metric("Consommation Totale (Période)", f"{total_consumption:,.0f} kWh")
col2.metric("Coût Énergétique Total", f"{total_cost:,.2f} OMR")
col3.metric("Température Moyenne", f"{avg_temp:.1f} °C")
col4.metric("Taux de Production Moyen", f"{avg_production:.1f} %")

# --- Section de Prédiction ---
st.header("🔮 Simulateur de Coût Énergétique")
st.markdown("Ajustez les paramètres pour prédire la consommation et le coût pour la prochaine heure.")

col1, col2 = st.columns(2)

with col1:
    temp_input = st.slider("Température Extérieure (°C)", min_value=10, max_value=50, value=30)
    prod_input = st.slider("Taux de Production (%)", min_value=0, max_value=100, value=80)

with col2:
    hour_input = st.selectbox("Heure de la Journée", options=range(24), format_func=lambda x: f"{x:02d}:00")
    dow_text_input = st.selectbox("Jour de la Semaine", options=["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
    dow_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, "Vendredi": 4, "Samedi": 5, "Dimanche": 6}
    dow_input = dow_map[dow_text_input]

# Déterminer le prix de l'énergie
if 8 <= hour_input <= 11 or 17 <= hour_input <= 21:
    price_input = 0.15 # Heures pleines
else:
    price_input = 0.08 # Heures creuses

# Créer l'input pour le modèle et prédire
input_data = pd.DataFrame([{
    'temperature': temp_input,
    'production_rate': prod_input,
    'hour': hour_input,
    'day_of_week': dow_input,
    'energy_price_omr_per_kwh': price_input
}])

predicted_consumption = predict_consumption(model, input_data)[0]
predicted_cost = predicted_consumption * price_input

st.subheader("Résultat de la Prédiction")
col1, col2 = st.columns(2)
col1.metric("Consommation Prédite", f"{predicted_consumption:.2f} kWh")
col2.metric("Coût Prédit", f"{predicted_cost:.4f} OMR")

# --- Graphiques ---
st.header("📈 Analyse Visuelle des Données")

# S'assurer que la colonne 'hour' existe
if 'hour' not in df.columns:
    df['hour'] = df['timestamp'].dt.hour

# Sélectionner les colonnes pour la corrélation
correlation_features = [
    'temperature', 
    'production_rate', 
    'energy_consumption_kwh', 
    'energy_price_omr_per_kwh', 
    'hour', 
    'day_of_week'
]
numeric_df = df[correlation_features]

# Graphique 1 : Consommation et Coût dans le temps
st.subheader("Consommation Énergétique et Coût Temps Réel")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['energy_consumption_kwh'], mode='lines', name='Consommation (kWh)', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['energy_consumption_kwh'] * df['energy_price_omr_per_kwh'], mode='lines', name='Coût (OMR)', yaxis='y2', line=dict(color='red')))
fig1.update_layout(
    xaxis_title='Date',
    yaxis=dict(title='Consommation (kWh)', side='left'),
    yaxis2=dict(title='Coût (OMR)', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig1, use_container_width=True)


# --- DÉBUT DE L'AJOUT ---

# Graphique 2 : Analyse des Corrélations
st.subheader("Matrice de Corrélation")

# --- LÉGENDE POUR LA MATRICE DE CORRÉLATION ---
st.markdown("""
### 📖 Comment lire cette matrice ?

Cette carte de chaleur montre les relations entre les différentes variables. Elle vous aide à comprendre ce qui influence la consommation énergétique.

*   **Les Couleurs :**
    *   **Rouge :** Corrélation **forte et positive**. Quand une variable augmente, l'autre a tendance à augmenter aussi. (Ex: Plus la production est élevée, plus la consommation est élevée).
    *   **Bleu :** Corrélation **forte et négative**. Quand une variable augmente, l'autre a tendance à diminuer.
    *   **Blanc :** Pas de corrélation linéaire évidente.

*   **Comment l'interpréter :**
    *   Regardez la ligne `energy_consumption_kwh` pour voir les facteurs qui influencent le plus votre consommation.
    *   Une corrélation proche de **+1 ou -1** est très forte. Proche de **0**, elle est faible.
""")

# Création et affichage du graphique
fig2 = go.Figure(data=go.Heatmap(
    z=numeric_df.corr(),
    x=numeric_df.columns,
    y=numeric_df.columns,
    colorscale='RdBu',
    zmid=0,
    text=numeric_df.corr().round(2), # Affiche la valeur de la corrélation dans chaque case
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps=False
))
fig2.update_layout(
    title='Carte de Chaleur des Corrélations',
    width=700,
    height=700
)
st.plotly_chart(fig2, use_container_width=True)

# --- FIN DE L'AJOUT ---
