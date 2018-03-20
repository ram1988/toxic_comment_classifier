import numpy as np
import nltk
from multiprocessing import Pool
import pickle
import re


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


from classifier import Classifier

class LogisticRegressor(Classifier):


    def __init__(self):
        self.logistic_classifier = pickle.load(open("data/logistic_model.pkl","rb"))
        self.bad_terms = self.read_bad_terms()

    def read_bad_terms(self):
        bad_list = open("data//bad_words","r")
        bad_list = str(bad_list.readlines()[0]).split(",")

        bad_terms = []
        for term in bad_list:
            print(term)
            bad_terms.append(term)

        bad_terms = list(set(bad_list))
        print(len(bad_terms))

        return bad_terms

    def train_model(self,trained_features,targets):
        self.logistic_classifier = []

        parameters = {
            'tfidf__use_idf': (True,False),
            'logis__C':(0.001,0.005)
        }

        for i,target in enumerate(targets):
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(vocabulary=self.bad_terms)),
                ('logis', LogisticRegression()),
            ])

            print("Training the model-->"+str(i))
            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

            grid_search.fit(trained_features,target)
            self.logistic_classifier.append(grid_search.best_estimator_)
            print("Best score: %0.3f" % grid_search.best_score_)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))


        return self.logistic_classifier

    def predict(self,test_features):
        test_outputs = []

        for test_vec in test_features:
            predictions = []
            for i,logis in enumerate(self.logistic_classifier):
                pred = logis.predict_proba([test_vec])
                rounded_pred = pred[0][1]
                rounded_pred = round(rounded_pred,2)
                predictions.append(rounded_pred)
            test_outputs.append(predictions)

        return test_outputs


#logis = LogisticRegressor()
#print(logis.read_bad_terms())
