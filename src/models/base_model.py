from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def predict_score():
        pass

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def save_model():
        pass

    @abstractmethod
    def load_model():
        pass

    @abstractmethod
    def instance_explanation():
        pass

    @abstractmethod
    def global_explanation():
        pass
