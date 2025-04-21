from typing import List, Optional, Tuple

from . import ModelABC


class CustomModel(ModelABC):
    aliases = "custom"  # Do not edit this

    def __init__(self, datafile: Optional[str] = None):
        """
        Initialize the custom model. If datafile is provided, load the model from the file.
        """
        pass  # Define This

    def predict(self, xseq: List[List[str]]) -> List[str]:
        """
        Predict the labels for the given sequences.
        """
        return ["O"] * len(xseq)  # Define This

    def fit(self, xseq: List[List[str]], yseq: List[str], classes: Optional[List[str]] = None, **kwargs) -> None:
        """
        Partial fit the model with new data.
        """
        pass  # Define This

    def save(self, model_file: str) -> None:
        """
        Save the model to a file.
        """
        pass  # Define This
