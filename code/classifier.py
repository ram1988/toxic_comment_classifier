from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def train_model(self,trained_features,targets):
        pass

    @abstractmethod
    def predict(self,test_features):
        pass
