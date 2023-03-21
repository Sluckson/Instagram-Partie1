



##I. Introduction
##II. Collecte et préparation des données
##III. Techniques de traitement du langage naturel
##IV. Expérimentation de différents algorithmes de classification
##V. Sélection du modèle final
##VI. Évaluation du modèle
##VII. Conclusion


###Impportation tous les packages necessaires
'''
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import gradio as gr
import warnings
import os
warnings.filterwarnings("ignore")
'''
###lire les donnees
df = pd.read_csv("/Users/surprice/data/jigsaw-toxic-comment-classification-challenge/train.csv/train.csv")

##Nettoyer les commentaires
'''
def clean_comments(text):
    text = text.lower()
    
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"\'s","",text)
    text = re.sub(r"n\'t","not",text)
    text = re.sub(r"","",text)
    text = re.sub(r"\'re","are",text)
    text = re.sub(r"i'm,","i am",text)
    text = re.sub(r"\'re","are",text)
    text = re.sub(r"\'d","would",text)
    text = re.sub(r"\'ll","will",text)
    text = re.sub(r"\'scuse","excuse",text)
    text = re.sub(r"\'t","not",text)
    
    
    
    text = re.replace("\\r"," ")
    text = re.replace("\\n"," ")
    text = re.replace("\\"," ")
    text = re.sub("[^A-Za-z0-9]+"," ",text)
    
    text = text.strip('')
    
    return text
'''

##visualisation des donnees
'''
comments_sum = len(df)
toxic_sum = df[toxicity_criteria[0]].sum()
severe_toxic_sum = df[toxicity_criteria[1]].sum()
obscene_sum = df[toxicity_criteria[2]].sum()
thread_sum = df[toxicity_criteria[3]].sum()
insult_sum = df[toxicity_criteria[4]].sum()
identity_hate_sum = df[toxicity_criteria[5]].sum()
'''

Collecte et préparation des données
La première étape dans l'utilisation du machine learning pour classifier les commentaires toxiques est la collecte et la préparation des données. Un jeu de données annotées de commentaires toxiques est nécessaire pour entraîner le modèle de classification. Le jeu de données "Toxic Comment Classification Challenge" a été utilisé pour ce projet. Les données ont été préparées en les nettoyant, en les normalisant et en les vectorisant.

##processus d'entrainement  des donnees
'''
com_batch_x, com_batch_y = comment_data.as_numpy_iterator().next()
print(com_batch_x.shape)
training_data = comment_data.take(int(len(comment_data)*0.7))
validating_data = comment_data.skip(len(training_data)).take(int(len(comment_data)*0.2))
testing_data = comment_data.skip(len(training_data)* len(validating_data)).take(int(len(comment_data)*0.1))
print(len(training_data))
'''

##construction le model sequentiel

'''
def create_sequencetial_model():
    seq = Sequential()
    seq.add(Embedding(300001, 32))
    seq.add(Bidirectional(LSTM(32,activation='tanh')))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dense(256, activation='relu'))
    seq.add(Dense(128, activation='relu'))
    
    seq.add(Dense(6, activation='sigmoid'))
    seq.compile(loss='BinaryCrossentropy', optimizer='Adam',metrics=['accuracy'])
    return seq
'''

##preduction du model
'''
test_comment = tokenizer('I hate you,you fat pig!')
test = model.predict(np.array([test_comment]))
toxicity_criteria
test
(test > 0.5).stype(bool)
com_batch_x, com_batch_y = comment_data.as_numpy_iterator().next()
test_data_model = model.predict(com_batch_x)
(test > 0.5).stype(bool)
'''

##evaluation du model
'''
model_score = model.avaluate(training_data, verbose=0)
print('Model loss:',model_score[0])
print('Model accuracy:',model_score[1])
'''
##enregistre et tester le model
'''
model.save('toxic_omment_etector.h5')
model =tf.keras.model.load_model('toxic_omment_etector.h5')
test_save_model = model.predict(np.array([test_comment]))
'''

##CREATION D'INTERFACE UTILISATEUR
'''
def evaluate_comment():
    tokenizer_comment = tokenizer([comment])
    result = model.predict(tokenizer_comment)
    text = 'Taper votre text:\n'
    for index, colums in enumerate(df[toxicity_criteria]):
        if colums == 'toxic':
            if result[0][index] > 0.5:
                text += '-toxic\n'
            else:
                text += '-non_toxic\n'
        elif colums == 'severe_toxic':
            if result[0][index] > 0.5:
                text += '-severe Toxic'
        elif colums == 'obscene':
            if result[0][index] > 0.5:
                text += '-Obscene\n'
        elif colums == 'threat':
            if result[0][index] > 0.5:
                text += '-threatening\n'
        elif colums == 'insult':
            if result[0][index] > 0.5:
                text += '-Offensive\n'
        elif colums == 'identity_hate':
            if result[0][index] > 0.5:
                text += '-racist\n' 
    if result[0][0]>0.5:
        text += "\n\nNo need to be toxic ! being nice won't hurt you \N{grimacing face}"
    else:
        text += "\n\nKeep it up \N{grimacing face}"
    return text
        
GradioGUI = gr.Interface(
fn = evaluate_comment,
inputs = gr.inputs.Textbox(lines=5,placeholder="Entrer votre commentaire"),
outputs="text",
title="Toxic detector comment",
description="Notre projet essaie de detecter les mauvais commentaires",
css='''span{text-transform: uppercase} p{text-align: center}''')
GradioGUI.launch(share=True)
'''


