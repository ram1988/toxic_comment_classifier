import numpy as np
import pandas as pd
import spacy
import pickle
import re

from nltk.corpus import stopwords

from logistic_regression import LogisticRegressor

nlp = spacy.load("en")
stopWords = set(stopwords.words('english'))

def loadQuestionsFromTrainDF():
    df = pd.read_csv("data\\train.csv")
    return df["comment_text"],df["toxic"],df["severe_toxic"],df["obscene"],df["threat"],df["insult"],df["identity_hate"]

def loadQuestionsFromTestDF():
    df = pd.read_csv("data\\test.csv")
    return df["id"],df["comment_text"]


def lemmatize(text):
    token = nlp(text)
    lemmatized_token = " ".join([tok.lemma_ if tok.lemma_ != '-PRON-' else str(tok) for tok in token])
    return lemmatized_token

def remove_stop_words(comments):
    cleaned_comments = []

    for i,text in enumerate(comments):
        #print(i)
        text = text.split()
        filtered_text = []
        for tok in text:
            if tok not in stopWords:
                filtered_text.append(tok)
        joined_text = " ".join(filtered_text)
        cleaned_comments.append(joined_text)

    return cleaned_comments

def lemmatize_text(comments):
    lemmatized_comments = []

    for i, text in enumerate(comments):
        print(i)
        if i != 46573:
            lemmatized_token = lemmatize(text)
            lemmatized_comments.append(lemmatized_token)

    print(len(lemmatized_comments))
    return lemmatized_comments


def lemmatize_train_text():
    comments, toxic_sets, severe_toxic_sets, obscene_sets, threat_sets, insult_sets, identity_hate_sets = loadQuestionsFromTrainDF()

    lemmatized_comments = remove_stop_words(comments)

    print(len(lemmatized_comments))
    comments = []
    for comment in lemmatized_comments:
        comment = str(comment).lower()
        comment = re.sub(r'[^\w\s]', '', comment)
        comments.append(comment)

    print(len(comments))

    dataframe = (comments, toxic_sets, severe_toxic_sets, obscene_sets, threat_sets, insult_sets, identity_hate_sets)
    pickle.dump(dataframe, open("lemmatized_train_dataframe.pkl","wb"))

def lemmatize_test_text():
    comments = loadQuestionsFromTestDF()[1]
    lemmatized_comments = lemmatize_text(comments)
    return

#train without stopwords with non-lemmatized text
def train_model():
    training_data = pd.read_pickle("..//data//lemmatized_train_dataframe.pkl")

    train_features = []
    for feature in training_data[0]:
        feature = feature.lower()
        train_features.append(feature)

    targets = []
    targets.append(training_data[1])
    targets.append(training_data[2])
    targets.append(training_data[3])
    targets.append(training_data[4])
    targets.append(training_data[5])
    targets.append(training_data[6])

    logis = LogisticRegressor()
    logis = logis.train_model(train_features,targets)

    pickle.dump(logis,open("../data/logistic_model.pkl","wb"))

def predict():
    ids, comments = loadQuestionsFromTestDF()
    logis = LogisticRegressor()
    final_submission = open("final.csv","w+")

    for i,text in enumerate(comments):
        print(ids[i])
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = remove_stop_words([text.lower()])
        text = lemmatize(text[0])
        results = logis.predict([text])[0]

        line = str(ids[i])+","
        for res in results:
            line += str(round(res,2))+","

        line = line[0:len(line)-1]
        print(line)
        final_submission.write(line+"\n")

#lemmatize_train_text()
#train_model()
predict()



