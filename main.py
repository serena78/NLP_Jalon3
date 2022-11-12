import streamlit as st
import pandas as pd
from predict import *
from PIL import Image

# Presentation de l'application

st.title('ApplicationNLP : RevieWAnalyzer')
myImage = Image.open("5-star-reviews-checked-PRE-1 (1).jpg")
myImage.show()

monlabel = "Quel texte analyser ? "
options = pd.DataFrame(['Avis dataset', 'Texte libre'])


n_topics = st.number_input(label= "Le nombre de topics", min_value=0, max_value=15)
with st.sidebar:
        st.radio(monlabel, options)
        text=st.text_input(label="Donnez nous votre avis, ça nous intéresse!")
if st.button(label = "Détecter le sujet d'insatisfaction") == True :
    pred=prediction(model_pred, vectorizer, n_topics, text)
    st.write(str(pred))



