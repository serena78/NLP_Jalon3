import pandas as pd
from pickle import *


#Pré-traitement 


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import contractions
tokenizer = RegexpTokenizer(r'\w+')

def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed



#import en_core_web_sm
#nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()
    
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'a')) # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'v')) # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'n')) # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'r')) # Lemmatise adverbs
        else:
            lemmatized_text_list.append(lemmatizer.lemmatize(word)) # If no tags has been found, perform a non specific lemmatisation
    
    return " ".join(lemmatized_text_list)




def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])
file_name1 = open("vectorizer (1).pkl",'rb')
vectorizer = load(file_name1)


def contraction_text(text):
    return contractions.fix(text)

negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"

def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i+1 for i in range(len(tokens)-1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx]= negative_prefix + tokens[idx]
    
    tokens = [token for i,token in enumerate(tokens) if i+1 not in negative_idx]
    
    return " ".join(tokens)


from spacy.lang.en.stop_words import STOP_WORDS

def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]
    
    return " ".join([word for word in text.split() if word not in english_stopwords])


def preprocess_text(text):
    
    # Tokenize review
    text = tokenize_text(text)
    
    # Lemmatize review
    text = lemmatize_text(text)
    
    # Normalize review
    text = normalize_text(text)
    
    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    return text


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')




file_name2 = open("nmf_model (1).pkl","rb")
model_pred = load(file_name2)
#print(vectorizer)
#print(model_pred)

from textblob import TextBlob
import numpy as np
def prediction(model, vectorizer, n_topic, new_reviews):
    new_reviews=preprocess_text(new_reviews)
    blob = TextBlob(new_reviews)
    sentimentBlob = blob.sentiment.polarity
    new_reviews = [new_reviews]
    new_reviews_transformed = vectorizer.transform(new_reviews)

    prediction = model.transform(new_reviews_transformed)

    topics = ['Esthétique du cadre', 'Qualité de la sauce', 'Qualité de la pizza',
              'Qualité du service au niveau de la prise de commande', 'Rapidité du sercive', 'Staff du restaurant',
              'Description des burgers',
              "Temps d'attente", "Menu poulet", "Boissons disponibles au bar",
              "Comparaison par rapport à un autre passage dans le restaurant", "Plainte des clients au manager",
              "Garniture sandwich", "Menu sushi",
              "Probabilité de revenir dans le restaurant"]
    if sentimentBlob < 0 and sentimentBlob > -1:

        max = np.argsort(prediction)
        max_list = (list(max[0]))
        max_list.reverse()
        print(max_list)
        topic = []
        for i in range(n_topic):
            topic.append(topics[max_list[i]])
        return sentimentBlob, prediction, topic

    return sentimentBlob

