from abc import ABC, abstractmethod


class ExportableModel(ABC):

    @abstractmethod
    def export(self, weights):
        pass


class SupervisedModel(ABC):

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def test(self, X):
        pass


def func():
    return 0
