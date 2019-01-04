from abc import ABC, abstractmethod

class AbstractTrainer(ABC):
    @abstractmethod
    def train (iterator):
        pass