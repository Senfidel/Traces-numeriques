import streamlit as st
import pandas as pd
import joblib
import gdown
import plotly.express as px
from bunkatopics import Bunka
from langchain_community.embeddings import HuggingFaceEmbeddings

# Télécharger le modèle depuis Google Drive
@st.cache_resource
def download_model(url, output):
    gdown.download(url, output, quiet=False)
    return output

# URL de téléchargement direct de Google Drive
url = 'https://drive.google.com/uc?id=1hEJdBpJuJJgMaOoS7NQ4PnGlftWkpbkK' 
output = 'model_bunka.pkl'
# Chemin du fichier modèle téléchargé
model_path = download_model(url, output)
bunka = joblib.load(model_path)

# Chargement des données
df_initial = pd.read_csv("df_youtube_final.csv")

# Titres
st.title("**Projet Traces Numériques**")
st.write("#### **M2 D2SN**")

st.markdown("")
st.markdown("")
# Sous-titre
st.markdown("### ***Régime Carnivore***")
st.markdown("")
st.markdown("***1. Présentation des données youtube***")

if st.checkbox(" **Données:** "):
    st.write(df_initial)

if st.checkbox("**Dimensions du dataframe :**"):
    st.write(df_initial.shape)

st.markdown("")
st.markdown("***2. Analyse des topics***")

fig = bunka.visualize_topics(width=800, height=800, colorscale='Portland', density=True, label_size_ratio=60, convex_hull=True)
# Vérifier et supprimer les annotations ou titres indésirables
fig.update_layout(title_text='', annotations=[])

topic_dist=bunka.df_topics_
if st.checkbox(" **Distribution des topics :** "):
    st.write(topic_dist)

if st.checkbox("**Espace des topics :** "):
    st.plotly_chart(fig)

top_docs_per_topic=bunka.df_top_docs_per_topic_
if st.checkbox(" **Tops vidéos par topic :** "):
    st.write(top_docs_per_topic)





