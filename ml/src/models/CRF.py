import sys
from itertools import batched
from tempfile import mktemp
from typing import List, Optional

import pycrfsuite

from . import ModelABC


class CRF(ModelABC):
    aliases = "crf"

    def __init__(self, datafile: Optional[str] = None):
        # Create a CRF Tagger object, and load given model
        if datafile is not None:
            self.weights = datafile
        else:
            self.weights = mktemp(suffix=".crf")

    def predict(self, xseq: List[List[str]]) -> List[str]:
        tagger = pycrfsuite.Tagger()
        tagger.open(self.weights)
        return tagger.tag(xseq)

    def fit(self, xseq: List[List[str]], yseq: List[str], classes: List[str] | None = None, **kwargs) -> None:
        trainer = pycrfsuite.Trainer()
        for x, y in zip(batched(xseq, 100), batched(yseq, 100)):
            trainer.append(x, y, 0)

        # Use L2-regularized SGD and 1st-order dyad features.
        trainer.select("l2sgd", "crf1d")

        # This demonstrates how to list parameters and obtain their values.
        trainer.set("feature.minfreq", 1)  # mininum frequecy of a feature to consider it
        trainer.set("c2", 0.1)  # coefficient for L2 regularization

        print("Training with following parameters: ")
        for name in trainer.params():
            print(name, trainer.get(name), trainer.help(name), file=sys.stderr)

        # Start training and dump model to modelfile
        trainer.train(self.weights)

    def save(self, model_file: str) -> None:
        """
        Save the model to a file.
        Args:
            model_file (str): The file where the trained model will be saved.
        """
        with open(model_file, "wb") as f, open(self.weights, "rb") as src:
            f.write(src.read())
