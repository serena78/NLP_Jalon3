import pandas as pd
from pickle import *
file_name1 = open("vectorizer (1).pkl",'rb')
vectorizer = load(file_name1)

file_name2 = open("nmf_model (1).pkl","rb")
model_pred = load(file_name2)
print(vectorizer)
print(model_pred)



def prediction(model, vectorizer, n_topic, new_reviews):
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

