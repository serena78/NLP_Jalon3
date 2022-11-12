import streamlit as st
import pandas as pd
from predict import *

# Presentation de l'application

st.title('Application : RevieWAnalyzer')

monlabel = "Quel texte analyser ? "
options = pd.DataFrame(['Avis dataset', 'Texte libre'])


n_topics = st.number_input(label= "Le nombre de topics", min_value=0, max_value=15)
with st.sidebar:
        st.radio(monlabel, options)
        text=st.text_input(label="Donnez nous votre avis")
if st.button(label = "Détecter le sujet d'insatisfaction") == True :
    pred=prediction(model_pred, vectorizer, n_topics, text)
    st.write(str(pred))



