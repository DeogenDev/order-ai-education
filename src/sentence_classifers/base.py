"""Базовый класс"""

from abc import ABC, abstractmethod

class SentenceClassifierlBase(ABC):

    @abstractmethod
    def run(self):
        pass
