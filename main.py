import streamlit as st
import pandas as pd
from predict import *

# Presentation de l'application

st.title('Application : RevieWAnalyzer')

monlabel = "Quel texte analyser ? "
options = pd.DataFrame(['Avis dataset', 'Texte libre'])

st.number_input(label= "Le nombre de topics", min_value=0, max_value=15)

if st.button(label = "Détecter le sujet d'insatisfaction") == True :
    prediction(model, vectorizer, n_topics, text)


with st.sidebar:
        st.radio(monlabel, options)
        st.text_input(label="Donnez nous votre avis")
