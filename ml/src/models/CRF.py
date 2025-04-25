import sys
from typing import List, Optional, Iterable, Iterator, TypeVar

import pycrfsuite

from . import ModelABC

T = TypeVar('T')

def batched(iterable: Iterable[T], n: int) -> Iterator[List[T]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

class CRF(ModelABC):
    aliases = "crf"

    def __init__(self, datafile: Optional[str] = None):
        if datafile is not None:
            self.weights = datafile
        else:
            from tempfile import mktemp
            self.weights = mktemp(suffix=".crf")

    def predict(self, xseq: List[List[str]]) -> List[str]:
        tagger = pycrfsuite.Tagger()
        tagger.open(self.weights)
        return tagger.tag(xseq)

    def fit(self, xseq: List[List[str]], yseq: List[str], classes: Optional[List[str]] = None, **kwargs) -> None:
        trainer = pycrfsuite.Trainer()
        for x, y in zip(batched(xseq, 100), batched(yseq, 100)):
            trainer.append(x, y, 0)

        trainer.select("l2sgd", "crf1d")

        trainer.set("feature.minfreq", 1)
        trainer.set("c2", 0.1)

        print("Training with following parameters: ", file=sys.stderr)
        for name in trainer.params():
            print(name, trainer.get(name), trainer.help(name), file=sys.stderr)

        trainer.train(self.weights)

    def save(self, model_file: str) -> None:
        with open(model_file, "wb") as f, open(self.weights, "rb") as src:
            f.write(src.read())
