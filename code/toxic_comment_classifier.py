import numpy as np
import pandas as pd
import spacy
import pickle

from logistic_regression import LogisticRegressor

nlp = spacy.load("en")

def loadQuestionsFromTrainDF():
    df = pd.read_csv("..\\data\\train.csv")
    return df["comment_text"],df["toxic"],df["severe_toxic"],df["obscene"],df["threat"],df["insult"],df["identity_hate"]

def loadQuestionsFromTestDF():
    df = pd.read_csv("..\\data\\test.csv")
    return df["id"],df["comment_text"]


def lemmatize_text(comments):
    lemmatized_comments = []

    for i, text in enumerate(comments):
        print(i)
        if i != 46573:
            token = nlp(text)
            lemmatized_token = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else str(tok) for tok in token])
            lemmatized_comments.append(lemmatized_token)

    print(len(lemmatized_comments))
    return lemmatized_comments


def lemmatize_train_text():
    comments, toxic_sets, severe_toxic_sets, obscene_sets, threat_sets, insult_sets, identity_hate_sets = loadQuestionsFromTrainDF()

    lemmatized_comments = lemmatize_text(comments)

    dataframe = (lemmatized_comments, toxic_sets, severe_toxic_sets, obscene_sets, threat_sets, insult_sets, identity_hate_sets)
    pickle.dump(dataframe, open("lemmatized_train_dataframe.pkl","wb"))

def lemmatize_test_text():
    comments = loadQuestionsFromTestDF()[1]
    lemmatized_comments = lemmatize_text(comments)
    pickle.dump(lemmatized_comments, open("lemmatized_test_dataframe.pkl","wb"))

def train_model():
    training_data = pd.read_pickle("../data/lemmatized_dataframe.pkl")

    train_features = training_data[0]

    targets = []
    targets.append(training_data[1])
    targets.append(training_data[2])
    targets.append(training_data[3])
    targets.append(training_data[4])
    targets.append(training_data[5])
    targets.append(training_data[6])

    logis = LogisticRegressor()
    logis = logis.train_model(train_features,targets)

    pickle.dump(logis,open("logistic_model.pkl","wb"))

#46573-to be done since it was big doc
#lemmatize_test_text()

train_model()





