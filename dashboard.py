import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from io import StringIO
shap.initjs()
import os
import streamlit as st

port = int(os.getenv("PORT", 8501))

################ FONCTIONS DE CACHE ################
# Scaler
@st.cache_data
def charger_scaler(scaler_path):
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du scaler : {e}")
        return None
    


# Modele
@st.cache_data
def charger_modele(model_path):
    try:
        pipeline = joblib.load(model_path)
        return pipeline.named_steps["classification"]
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Predictions
@st.cache_data
def obtenir_predictions(uploaded_file):
    response = requests.post(api_url, files={"file": uploaded_file.getvalue()})
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        st.error("Erreur dans l'appel API.")
        return pd.DataFrame()

# Jauge prédiction
def afficher_jauge(probabilite):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilite * 100,
        title={'text': "Scoring du client"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "white"},
               'steps': [
                   {'range': [0, seuil*100], 'color': "green"},
                   {'range': [seuil*100, 100], 'color': "red"}]
        }))
    st.plotly_chart(fig)



# Configuration de la page
st.set_page_config(
    page_title="Dashboard Crédit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le scaler et le modèle
scaler_path = "models/scaler.pkl"
model_path = "models/model.pkl"
scaler = charger_scaler(scaler_path)
model = charger_modele(model_path)

# URL de l'API
api_url = "https://trvdln-credit-scoring-6d229fd55df0.herokuapp.com/score"
seuil = 0.54  # Seuil de classification

# Interface
st.title("🔍 Analyse des scores de crédit")
st.sidebar.header("📂 Options")
uploaded_file = st.sidebar.file_uploader("Importer un fichier CSV", type=["csv"], help="Chargez un fichier contenant les données des clients.")

# Session state pour conserver les états
if "afficher_feature_importance" not in st.session_state:
    st.session_state.afficher_feature_importance = False
afficher_feature_importance = st.sidebar.checkbox("Afficher l'importance des features", value=st.session_state.afficher_feature_importance)
st.session_state.afficher_feature_importance = afficher_feature_importance

# afficher_feature_importance = st.sidebar.checkbox("Afficher l'importance des features", value=False)
afficher_distributions = st.sidebar.checkbox("Afficher les distributions", value=False)
afficher_analyse = st.sidebar.checkbox("Afficher l'analyse bi-variée", value=False)

if uploaded_file:
    st.sidebar.success("✅ Fichier chargé avec succès")
    df = pd.read_csv(uploaded_file)
    
    # Envoyer le fichier à l'API
    predictions = obtenir_predictions(uploaded_file)
    if not predictions.empty:
        st.subheader("📊 Résultats des Prédictions")
        
        # Sélection d'un client
        client_id = st.selectbox("Sélectionnez un ID client :", predictions["Client_ID"], help="Choisissez un client pour voir son analyse détaillée.")
        client_data = predictions[predictions["Client_ID"] == client_id]
        
        if not client_data.empty:
            # Informations générales du client
            st.markdown(f"### 🏷️ Informations du client ID : **{client_id}**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                age = df.loc[df["SK_ID_CURR"] == client_id, "DAYS_BIRTH"].values[0]
                st.metric("📅 Âge", f"{int(-age/365)} ans")
            
            with col2:
                nb_child = df.loc[df["SK_ID_CURR"] == client_id, "CNT_CHILDREN"].values[0]
                st.metric("👶 Nombre d'enfants", f"{nb_child}")
            
            with col3:
                salary = df.loc[df["SK_ID_CURR"] == client_id, "AMT_INCOME_TOTAL"].values[0]
                st.metric("💰 Salaire", f"{salary:.0f} €")
            
            with col4:
                employ_seniority = df.loc[df["SK_ID_CURR"] == client_id, "DAYS_EMPLOYED"].values[0]
                st.metric("🏢 Ancienneté", f"{int(-employ_seniority/365)} ans")
            
            # Probabilité de défaut
            probabilite = client_data['Probabilite_Classe_1'].values[0]
            if probabilite > seuil:
                st.error(f"🚨 **Le modèle préconise de ne pas accorder le crédit.**")
            else:
                st.success(f"✅ **Le modèle ne détecte pas de risque de défauts de paiments.**")
            # Affichage de la jauge
            afficher_jauge(probabilite)
            
            
            
            # Explication des prédictions
            if model and scaler:
                X = df.drop(["SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "CNT_CHILDREN"], axis=1)
                X_scaled = scaler.transform(X)
                df_importance = pd.DataFrame(X_scaled, columns=X.columns)
                explainer = shap.LinearExplainer(model, df_importance)
                shap_values = explainer(df_importance)
                
                client_index = df.index[df["SK_ID_CURR"] == client_id].tolist()[0]
                
                if afficher_feature_importance:
                    fig_waterfall = plt.figure(figsize=(6, 4))
                    shap.waterfall_plot(shap_values[client_index])
                    st.subheader(f"📈 Analyse de l'influence des variables pour le client {client_id}")
                    st.pyplot(fig_waterfall)
                    st.subheader(f"📈 Analyse de l'influence des variables pour tous les clients")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X)
                    st.pyplot(fig)



                if afficher_distributions:
                    variable = st.selectbox("Choisissez une variable pour voir la distribution :", X.columns)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df['Classe_Predite'] = predictions['Classe_Predite']
                    
                    for classe in df['Classe_Predite'].unique():
                        subset = df[df['Classe_Predite'] == classe]
                        sns.kdeplot(subset[variable], ax=ax, label=f'Classe {classe}', fill=True, alpha=0.5)
                        # ax.hist(subset[variable], bins=30, alpha=0.5, label=f'Classe {classe}')
                    
                    valeur_client = df.loc[df["SK_ID_CURR"] == client_id, variable].values[0]
                    ax.axvline(valeur_client, color='red', linestyle='dashed', linewidth=2, label='Valeur du client')
                    ax.set_title(f"Distribution de {variable} selon la classe prédite")
                    ax.legend()
                    st.pyplot(fig)

                if afficher_analyse:
                    feature_x = st.selectbox("🛠️ Sélectionnez la 1ère feature :", X.columns, key="feature_x")
                    feature_y = st.selectbox("🛠️ Sélectionnez la 2ème feature :", X.columns, key="feature_y")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(X[feature_x], X[feature_y], alpha=0.6)
                    ax.set_xlabel(feature_x)
                    ax.set_ylabel(feature_y)
                    ax.set_title(f"Scatterplot de {feature_x} vs {feature_y}")
                    st.pyplot(fig)
        
        else:
            st.error("Aucune donnée trouvée pour ce client.")
else:
    st.warning("📥 Veuillez importer un fichier CSV avant de continuer.")
