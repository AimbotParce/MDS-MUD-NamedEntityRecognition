from typing import List, Optional, Tuple

from . import ModelABC


class CustomModel(ModelABC):
    aliases = "custom"  # Edit this

    def __init__(self, datafile: Optional[str] = None):
        """
        Initialize the custom model. If datafile is provided, load the model from the file.
        """
        raise NotImplementedError("Custom model loading is not implemented.")  # Define This

    def predict(self, xseq: List[List[str]]) -> List[str]:
        """
        Predict the labels for the given sequences.
        """
        raise NotImplementedError("Custom model prediction is not implemented.")  # Define This

    def fit(self, xseq: List[List[str]], yseq: List[str], classes: Optional[List[str]] = None, **kwargs) -> None:
        """
        Partial fit the model with new data.
        """
        raise NotImplementedError("Partial fit is not supported for the custom model.")  # Define this

    def save(self, model_file: str) -> None:
        """
        Save the model to a file.
        """
        raise NotImplementedError("Custom model saving is not implemented.")
