from abc import ABC, abstractmethod


class AbstractSupervisedModel(ABC):
    
    @abstractmethod
    def fit(X, y, **kwargs):
        pass
    
    @abstractmethod
    def predict(X, **kwargs):
        pass

class AbstractFrameworkModel(ABC):
    @abstractmethod
    def save(**kwargs):
        pass
    @abstractmethod
    def load(**kwargs):
        pass