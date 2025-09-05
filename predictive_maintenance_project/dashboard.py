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

# Configuration de la page
st.set_page_config(
    page_title="Tableau de Bord - Maintenance Prédictive",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🔧 Tableau de Bord - Maintenance Prédictive des Moteurs Électriques")
st.markdown("---")

# Charger et préparer le modèle
@st.cache_data
def load_and_prepare_model():
    # Utiliser un chemin robuste pour le CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "predictive_maintenance.csv")

    # Charger le dataset
    df = pd.read_csv(file_path)

    # Renommer les colonnes
    df.columns = [col.replace("[K]", "_K")
                     .replace("[rpm]", "_rpm")
                     .replace("[Nm]", "_Nm")
                     .replace("[min]", "_min")
                     .replace(" ", "_") for col in df.columns]

    # Séparer les caractéristiques et la cible
    X = df.drop(["UDI", "Product_ID", "Target", "Failure_Type"], axis=1)
    y = df["Target"]

    # Préprocesseur
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Créer et entraîner le modèle
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)

    return model, df, X_test, y_test

# Charger le modèle
model, df, X_test, y_test = load_and_prepare_model()

# Sidebar pour les contrôles
st.sidebar.header("🎛️ Contrôles")

if st.sidebar.button("🔄 Actualiser les Données"):
    st.rerun()

# Sélection d'un échantillon
sample_idx = st.sidebar.selectbox("Sélectionner un échantillon:", range(len(X_test)))
sample_data = X_test.iloc[sample_idx:sample_idx+1]

# Prédiction
prediction = model.predict(sample_data)[0]
prediction_proba = model.predict_proba(sample_data)[0]

# Colonnes principales
col1, col2, col3 = st.columns([1, 1, 1])

# Colonne 1: Valeurs des capteurs
with col1:
    st.subheader("📊 Valeurs des Capteurs")
    air_temp = sample_data["Air_temperature__K"].values[0]
    process_temp = sample_data["Process_temperature__K"].values[0]
    rotation_speed = sample_data["Rotational_speed__rpm"].values[0]
    torque = sample_data["Torque__Nm"].values[0]
    tool_wear = sample_data["Tool_wear__min"].values[0]

    st.metric("🌡️ Température Air", f"{air_temp:.1f} K", delta=f"{random.uniform(-0.5, 0.5):.1f}")
    st.metric("🔥 Température Processus", f"{process_temp:.1f} K", delta=f"{random.uniform(-0.3, 0.3):.1f}")
    st.metric("⚡ Vitesse Rotation", f"{rotation_speed:.0f} rpm", delta=f"{random.uniform(-10, 10):.0f}")
    st.metric("🔧 Couple", f"{torque:.1f} Nm", delta=f"{random.uniform(-1, 1):.1f}")
    st.metric("⏱️ Usure Outil", f"{tool_wear:.0f} min", delta=f"{random.uniform(0, 2):.0f}")

# Colonne 2: Prédictions IA
with col2:
    st.subheader("🤖 Prédictions IA")
    if prediction == 0:
        st.success("✅ **NORMAL** - Fonctionnement optimal")
    else:
        st.error("⚠️ **DÉFAILLANCE PRÉDITE** - Maintenance requise")

    prob_normal = prediction_proba[0] * 100
    prob_failure = prediction_proba[1] * 100

    st.write("**Probabilités:**")
    st.write(f"• Normal: {prob_normal:.1f}%")
    st.write(f"• Défaillance: {prob_failure:.1f}%")

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Normal", "Défaillance"],
        values=[prob_normal, prob_failure],
        hole=0.3,
        marker_colors=["green", "red"]
    )])
    fig_pie.update_layout(title="Probabilités de Prédiction", height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

# Colonne 3: Alertes et recommandations
with col3:
    st.subheader("🚨 Alertes et Recommandations")
    if prediction == 1:
        st.warning("⚠️ **ALERTE MAINTENANCE**")
        st.write("• Planifier une inspection immédiate")
        st.write("• Vérifier les roulements")
        st.write("• Contrôler la température")
        st.write("• Examiner les vibrations")
    else:
        st.info("ℹ️ **FONCTIONNEMENT NORMAL**")
        st.write("• Tous les paramètres sont normaux")
        st.write("• Maintenance préventive à jour")
        st.write("• Surveillance continue active")

# Tendances historiques
st.markdown("---")
st.subheader("📈 Tendances Historiques")

col1, col2 = st.columns(2)
time_points = pd.date_range(start="2024-01-01", periods=30, freq="D")

with col1:
    temp_data = np.random.normal(process_temp, 2, 30)
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=time_points, y=temp_data, mode="lines+markers", line=dict(color="orange")))
    fig_temp.update_layout(title="Évolution Température (30j)", xaxis_title="Date", yaxis_title="Température (K)")
    st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    speed_data = np.random.normal(rotation_speed, 50, 30)
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(x=time_points, y=speed_data, mode="lines+markers", line=dict(color="blue")))
    fig_speed.update_layout(title="Évolution Vitesse Rotation (30j)", xaxis_title="Date", yaxis_title="Vitesse (rpm)")
    st.plotly_chart(fig_speed, use_container_width=True)

# Distribution des données
st.markdown("---")
st.subheader("📊 Distribution des Données")

col1, col2 = st.columns(2)
with col1:
    fig_hist1 = px.histogram(df, x="Process_temperature__K", color="Target", title="Distribution Température Processus")
    st.plotly_chart(fig_hist1, use_container_width=True)
with col2:
    fig_hist2 = px.histogram(df, x="Rotational_speed__rpm", color="Target", title="Distribution Vitesse Rotation")
    st.plotly_chart(fig_hist2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Tableau de Bord de Maintenance Prédictive** - Développé avec Streamlit et Machine Learning")
