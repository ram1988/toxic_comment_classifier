from abc import ABCMeta, abstractmethod


class Classifier(ABCMeta):

    @abstractmethod
    def train_model(self,trained_features,targets):
        pass

    @abstractmethod
    def predict(self,test_features):
        pass