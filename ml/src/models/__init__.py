import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Type


class ModelABC(ABC):
    allowed_extensions: Tuple[str]

    @abstractmethod
    def __init__(self, datafile: str): ...
    @abstractmethod
    def predict(self, xseq: List[List[str]]) -> List[str]: ...

    def __init_subclass__(cls) -> None:
        # Register the subclass in the global list of model classes
        res = super().__init_subclass__()
        if not hasattr(cls, "allowed_extensions"):
            raise ValueError(f"Class {cls.__name__} must define allowed_extensions as a class variable.")

        global MODEL_CLASSES
        MODEL_CLASSES.append(cls)
        return res


MODEL_CLASSES: List[Type[ModelABC]] = []


class Model(object):
    """
    Class to load an arbitrary model and call it
    on a sequence.
    """

    def __init__(self, datafile: str):
        """
        Load a model from file.

        Args:
            datafile (str): The file where the model is stored.
        """
        for cls in MODEL_CLASSES:
            if os.path.splitext(datafile)[1] in cls.allowed_extensions:
                self._model = cls(datafile)
                break
        else:
            raise ValueError(f"Unknown model file extension: {os.path.splitext(datafile)[1]}")

    def predict(self, xseq: List[List[str]]) -> List[str]:
        return self._model.predict(xseq)


from .CRF import CRF
