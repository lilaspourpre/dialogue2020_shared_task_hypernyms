from abc import abstractmethod


class Model:
    @abstractmethod
    def predict_hypernyms(self, neologisms):
        pass
