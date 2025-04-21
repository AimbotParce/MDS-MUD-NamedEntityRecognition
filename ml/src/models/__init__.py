import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type


class ModelABC(ABC):
    aliases: Tuple[str, ...] | str | List[str] = []

    @abstractmethod
    def __init__(self, datafile: Optional[str] = None): ...
    @abstractmethod
    def predict(self, xseq: List[List[str]]) -> List[str]: ...
    @abstractmethod
    def fit(self, xseq: List[List[str]], yseq: List[str], classes: Optional[List[str]] = None, **kwargs) -> None: ...
    @abstractmethod
    def save(self, model_file: str) -> None: ...

    def __init_subclass__(cls) -> None:
        # Register the subclass in the global list of model classes
        res = super().__init_subclass__()
        if not hasattr(cls, "aliases"):
            raise ValueError(f"Class {cls.__name__} must define aliases as a class variable.")

        global MODEL_CLASSES
        names = [cls.__name__]
        if isinstance(cls.aliases, str):
            names.append(cls.aliases)
        elif isinstance(cls.aliases, (list, tuple)):
            names.extend(cls.aliases)
        else:
            raise TypeError(f"Class {cls.__name__} aliases must be a string or a list/tuple of strings.")
        for name in names:
            if name in MODEL_CLASSES:
                raise ValueError(f"Model class {name} already registered.")
            MODEL_CLASSES[name] = cls
        return res


MODEL_CLASSES: Dict[str, Type[ModelABC]] = {}


def load_model(datafile: Optional[str] = None, model_type: Optional[str] = None) -> ModelABC:
    """
    Load a model from file.

    Args:
        datafile (str): The file where the model is stored.
        model_type (str, optional): The type of model to load. If None, the type is inferred by trying to load the model
                                    with each registered class.
    """
    if datafile is None and model_type is None:
        raise ValueError("Either datafile or model_type must be specified.")
    if datafile is not None and not os.path.exists(datafile):
        raise ValueError(f"File {datafile} does not exist.")

    if model_type is not None:
        cls = MODEL_CLASSES.get(model_type)
        if cls is None:
            raise ValueError(f"Model type {model_type} not found.")
        return cls(datafile)
    else:
        for cls in set(MODEL_CLASSES.values()):
            try:
                loaded_model = cls(datafile)
            except:
                pass
            else:
                return loaded_model
        else:
            raise ValueError(f"Unable to load model from file {datafile}. No matching model class found.")


from .CRF import CRF
from .custom import CustomModel
from .multinomial_nb import MultinomialNB
