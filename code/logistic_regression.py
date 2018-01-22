import numpy as np
import nltk


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from classifier import Classifier

class LogisticRegression(Classifier):



    def __prepare_tfidf_vectorizer(self,text_features):
        self.vectorizer = TfidfVectorizer(max_features=50000, lowercase=True, analyzer='word',
                                    stop_words='english', ngram_range=(1, 3), dtype=np.float32)
        return self.vectorizer.fit_transform(text_features)


    def train_model(self,trained_features,targets):
        trained_vectors = self.__prepare_tfidf_vectorizer(trained_features)

        self.logistic_classifier = []

        for target in targets:
            logis = LogisticRegression(C=0.001)
            logis.fit(trained_vectors,target)
            self.logistic_classifier.append(logis)

    def predict(self,test_features):
        test_vectors = self.vectorizer.transform(test_features)

        test_outputs = []
        for test_vec in test_vectors:
            for i,logis in enumerate(self.logistic_classifier):
                predictions = {}
                pred = logis.predict(test_vec)
                predictions[i] = pred
                test_outputs.append(predictions)

        return test_outputs



