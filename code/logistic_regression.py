import numpy as np
import nltk
from multiprocessing import Pool
import pickle


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


from classifier import Classifier

class LogisticRegressor(Classifier):


    def __init__(self):
        self.logistic_classifier = pickle.load(open("../data/logistic_model.pkl","rb"))


    def train_model(self,trained_features,targets):
        self.logistic_classifier = []

        parameters = {
            'tfidf__use_idf': (True,False),
            'logis__C':(0.001,0.005)
        }

        for i,target in enumerate(targets):
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=50000)),
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
            for i,logis in enumerate(self.logistic_classifier):
                predictions = {}
                pred = logis.predict_proba([test_vec])
                print(pred)
                predictions[i] = pred
                test_outputs.append(predictions)

        return test_outputs



