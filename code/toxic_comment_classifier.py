import numpy as np
import pandas as pd
import spacy
import pickle

def loadQuestionsFromTrainDF():
    df = pd.read_csv("..\\data\\train.csv")
    return df["comment_text"],df["toxic"],df["severe_toxic"],df["obscene"],df["threat"],df["insult"],df["identity_hate"]

def loadQuestionsFromTestDF():
    df = pd.read_csv("..\\data\\test.csv")
    return df["id"],df["comment_text"]

def lemmatize_text():
    comments, toxic_sets, severe_toxic_sets, obscene_sets, threat_sets, insult_sets, identity_hate_sets = loadQuestionsFromTrainDF()
    nlp = spacy.load("en")

    lemmatized_comments = []
    for i,text in enumerate(comments):
        print(i)
        token = nlp(text)
        lemmatized_token = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else str(tok) for tok in token ])
        lemmatized_comments.append(lemmatized_token)

    print(len(lemmatized_comments))

    dataframe = (lemmatized_comments, toxic_sets, severe_toxic_sets, obscene_sets, threat_sets, insult_sets, identity_hate_sets)
    pickle.dump(dataframe, open("lemmatized_dataframe.pkl","wb"))

lemmatize_text()







