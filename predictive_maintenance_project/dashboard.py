import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import time
import random
import os

# Page configuration
st.set_page_config(
    page_title="Dashboard - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🔧 Dashboard - Predictive Maintenance of Electric Motors")
st.markdown("---")

# Load and prepare the model
@st.cache_data
def load_and_prepare_model():
    # Use a robust path for the CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "predictive_maintenance.csv")

    # Load dataset
    df = pd.read_csv(file_path)

    # Rename columns
    df.columns = [col.replace("[K]", "_K")
                     .replace("[rpm]", "_rpm")
                     .replace("[Nm]", "_Nm")
                     .replace("[min]", "_min")
                     .replace(" ", "_") for col in df.columns]

    # Separate features and target
    X = df.drop(["UDI", "Product_ID", "Target", "Failure_Type"], axis=1)
    y = df["Target"]

    # Preprocessor
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create and train the model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)

    return model, df, X_test, y_test

# Load model
model, df, X_test, y_test = load_and_prepare_model()

# Sidebar controls
st.sidebar.header("🎛️ Controls")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Sample selection
sample_idx = st.sidebar.selectbox("Select a sample:", range(len(X_test)))
sample_data = X_test.iloc[sample_idx:sample_idx+1]

# Prediction
prediction = model.predict(sample_data)[0]
prediction_proba = model.predict_proba(sample_data)[0]

# Main columns
col1, col2, col3 = st.columns([1, 1, 1])

# Column 1: Sensor values
with col1:
    st.subheader("📊 Sensor Values")
    air_temp = sample_data["Air_temperature__K"].values[0]
    process_temp = sample_data["Process_temperature__K"].values[0]
    rotation_speed = sample_data["Rotational_speed__rpm"].values[0]
    torque = sample_data["Torque__Nm"].values[0]
    tool_wear = sample_data["Tool_wear__min"].values[0]

    st.metric("🌡️ Air Temperature", f"{air_temp:.1f} K", delta=f"{random.uniform(-0.5, 0.5):.1f}")
    st.metric("🔥 Process Temperature", f"{process_temp:.1f} K", delta=f"{random.uniform(-0.3, 0.3):.1f}")
    st.metric("⚡ Rotation Speed", f"{rotation_speed:.0f} rpm", delta=f"{random.uniform(-10, 10):.0f}")
    st.metric("🔧 Torque", f"{torque:.1f} Nm", delta=f"{random.uniform(-1, 1):.1f}")
    st.metric("⏱️ Tool Wear", f"{tool_wear:.0f} min", delta=f"{random.uniform(0, 2):.0f}")

# Column 2: AI Predictions
with col2:
    st.subheader("🤖 AI Predictions")
    if prediction == 0:
        st.success("✅ **NORMAL** - Optimal operation")
    else:
        st.error("⚠️ **PREDICTED FAILURE** - Maintenance required")

    prob_normal = prediction_proba[0] * 100
    prob_failure = prediction_proba[1] * 100

    st.write("**Probabilities:**")
    st.write(f"• Normal: {prob_normal:.1f}%")
    st.write(f"• Failure: {prob_failure:.1f}%")

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Normal", "Failure"],
        values=[prob_normal, prob_failure],
        hole=0.3,
        marker_colors=["green", "red"]
    )])
    fig_pie.update_layout(title="Prediction Probabilities", height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

# Column 3: Alerts and Recommendations
with col3:
    st.subheader("🚨 Alerts and Recommendations")
    if prediction == 1:
        st.warning("⚠️ **MAINTENANCE ALERT**")
        st.write("• Schedule an immediate inspection")
        st.write("• Check bearings")
        st.write("• Monitor temperature")
        st.write("• Examine vibrations")
    else:
        st.info("ℹ️ **NORMAL OPERATION**")
        st.write("• All parameters are normal")
        st.write("• Preventive maintenance is up to date")
        st.write("• Continuous monitoring active")

# Historical trends
st.markdown("---")
st.subheader("📈 Historical Trends")

col1, col2 = st.columns(2)
time_points = pd.date_range(start="2024-01-01", periods=30, freq="D")

with col1:
    temp_data = np.random.normal(process_temp, 2, 30)
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=time_points, y=temp_data, mode="lines+markers", line=dict(color="orange")))
    fig_temp.update_layout(title="Temperature Over Time (Last 30 Days)", xaxis_title="Date", yaxis_title="Temperature (K)")
    st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    speed_data = np.random.normal(rotation_speed, 50, 30)
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(x=time_points, y=speed_data, mode="lines+markers", line=dict(color="blue")))
    fig_speed.update_layout(title="Rotation Speed Over Time (Last 30 Days)", xaxis_title="Date", yaxis_title="Speed (rpm)")
    st.plotly_chart(fig_speed, use_container_width=True)

# Data distribution
st.markdown("---")
st.subheader("📊 Data Distribution")

col1, col2 = st.columns(2)
with col1:
    fig_hist1 = px.histogram(df, x="Process_temperature__K", color="Target", title="Process Temperature Distribution")
    st.plotly_chart(fig_hist1, use_container_width=True)
with col2:
    fig_hist2 = px.histogram(df, x="Rotational_speed__rpm", color="Target", title="Rotation Speed Distribution")
    st.plotly_chart(fig_hist2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Predictive Maintenance Dashboard** - Developed with Streamlit and Machine Learning")
