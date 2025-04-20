#####################################################
## Class to store an ngram ME model
#####################################################

from typing import List

import pycrfsuite

from . import ModelABC


class CRF(ModelABC):
    allowed_extensions = (".crf",)

    def __init__(self, datafile: str):
        # Create a CRF Tagger object, and load given model
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(datafile)

    def predict(self, xseq: List[List[str]]) -> List[str]:
        return self.tagger.tag(xseq)
