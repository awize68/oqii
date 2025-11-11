# dashboard_en.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Cortex Energetics - OQII POC",
    page_icon="⚡",
    layout="wide"
)

# --- Utility Functions ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('usine_data.csv')
        st.success("Data file 'usine_data.csv' loaded successfully.")
        return df
    except FileNotFoundError:
        st.error("ERROR: The file 'usine_data.csv' was not found. Please make sure you have run data_generator.py first.")
        st.stop() # Stop script execution

@st.cache_resource
def load_model():
    try:
        model = joblib.load('energy_predictor.pkl')
        st.success("Model 'energy_predictor.pkl' loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("ERROR: The file 'energy_predictor.pkl' was not found. Please make sure you have run model.py first.")
        st.stop() # Stop script execution

def predict_consumption(model, input_df):
    return model.predict(input_df)

# --- Load Resources ---
df = load_data()
model = load_model()
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Dashboard Interface ---
st.title("⚡ Cortex Energetics - Operational Dashboard")
st.markdown("POC for OQ for Industrial Investments - Energy Cost Optimization")

# --- Top KPIs ---
st.header("📊 Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

total_consumption = df['energy_consumption_kwh'].sum()
total_cost = (df['energy_consumption_kwh'] * df['energy_price_omr_per_kwh']).sum()
avg_temp = df['temperature'].mean()
avg_production = df['production_rate'].mean()

col1.metric("Total Consumption (Period)", f"{total_consumption:,.0f} kWh")
col2.metric("Total Energy Cost", f"{total_cost:,.2f} OMR")
col3.metric("Average Temperature", f"{avg_temp:.1f} °C")
col4.metric("Average Production Rate", f"{avg_production:.1f} %")

# --- Prediction Section ---
st.header("🔮 Energy Cost Simulator")
st.markdown("Adjust the parameters to predict the consumption and cost for the next hour.")

col1, col2 = st.columns(2)

with col1:
    temp_input = st.slider("Outside Temperature (°C)", min_value=10, max_value=50, value=30)
    prod_input = st.slider("Production Rate (%)", min_value=0, max_value=100, value=80)

with col2:
    hour_input = st.selectbox("Hour of the Day", options=range(24), format_func=lambda x: f"{x:02d}:00")
    dow_text_input = st.selectbox("Day of the Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    dow_input = dow_map[dow_text_input]

# Determine energy price
if 8 <= hour_input <= 11 or 17 <= hour_input <= 21:
    price_input = 0.15 # Peak hours
else:
    price_input = 0.08 # Off-peak hours

# Create input for the model and predict
# The order of columns must be IDENTICAL to the one used for training
input_data = pd.DataFrame([{
    'temperature': temp_input,
    'production_rate': prod_input,
    'hour': hour_input,
    'day_of_week': dow_input, # <-- Use the number here
    'energy_price_omr_per_kwh': price_input
}])

predicted_consumption = predict_consumption(model, input_data)[0]
predicted_cost = predicted_consumption * price_input

st.subheader("Prediction Result")
col1, col2 = st.columns(2)
col1.metric("Predicted Consumption", f"{predicted_consumption:.2f} kWh")
col2.metric("Predicted Cost", f"{predicted_cost:.4f} OMR")

# --- Visual Analysis ---
st.header("📈 Visual Data Analysis")

# Ensure the 'hour' column exists for the correlation graph
if 'hour' not in df.columns:
    df['hour'] = df['timestamp'].dt.hour

# Select only numeric columns for correlation
correlation_features = [
    'temperature', 
    'production_rate', 
    'energy_consumption_kwh', 
    'energy_price_omr_per_kwh', 
    'hour', 
    'day_of_week'
]
numeric_df = df[correlation_features]

# Graph 1: Consumption and Cost over time
st.subheader("Real-Time Energy Consumption and Cost")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['energy_consumption_kwh'], mode='lines', name='Consumption (kWh)', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=df['timestamp'], y=df['energy_consumption_kwh'] * df['energy_price_omr_per_kwh'], mode='lines', name='Cost (OMR)', yaxis='y2', line=dict(color='red')))
fig1.update_layout(
    xaxis_title='Date',
    yaxis=dict(title='Consumption (kWh)', side='left'),
    yaxis2=dict(title='Cost (OMR)', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig1, use_container_width=True)

# Graph 2: Correlation Analysis
st.subheader("Correlation Matrix")

# --- LEGEND FOR THE CORRELATION MATRIX ---
st.markdown("""
### 📖 How to Read This Matrix?

This heatmap shows the relationships between different variables. It helps you understand what influences energy consumption.

*   **The Colors:**
    *   **Red:** A **strong positive** correlation. When one variable goes up, the other tends to go up as well. (e.g., The higher the production, the higher the consumption).
    *   **Blue:** A **strong negative** correlation. When one variable goes up, the other tends to go down.
    *   **White:** No obvious linear correlation.

*   **How to Interpret It:**
    *   Look at the `energy_consumption_kwh` row to see the factors that most influence your consumption.
    *   A correlation close to **+1 or -1** is very strong. Close to **0**, it is weak.
""")

# Create and display the graph
fig2 = go.Figure(data=go.Heatmap(
    z=numeric_df.corr(),
    x=numeric_df.columns,
    y=numeric_df.columns,
    colorscale='RdBu',
    zmid=0,
    text=numeric_df.corr().round(2), # Display the correlation value in each cell
    texttemplate="%{text}",
    textfont={"size": 10},
    hoverongaps=False
))
fig2.update_layout(
    title='Correlation Heatmap',
    width=700,
    height=700
)
st.plotly_chart(fig2, use_container_width=True)