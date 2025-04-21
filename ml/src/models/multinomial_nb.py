import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import dump, load
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB

from . import ModelABC


class MultinomialNB(ModelABC):
    aliases = "mnb", "multinomial_nb", "sklearn_mnb"

    def __init__(self, datafile: Optional[str] = None):
        """
        Initialize the custom model. If datafile is provided, load the model from the file.
        """
        if datafile is not None:
            self.mnb = load(datafile)
            self.vectorizer = load(self._get_vec_filename(datafile))
        else:
            self.vectorizer = DictVectorizer()
            self.mnb = SklearnMultinomialNB()

    def predict(self, xseq: List[List[str]]) -> List[str]:
        """
        Predict the labels for the given sequences.
        """
        xseq_array = np.asarray(list(map(self._to_dict, xseq)))
        X = self.vectorizer.transform(xseq_array)
        return self.mnb.predict(X)

    def fit(self, xseq: List[List[str]], yseq: List[str], **kwargs) -> None:
        """
        Fit the model with new data.
        """
        xseq_array = np.asarray(list(map(self._to_dict, xseq)))
        unique_classes = np.unique(yseq)
        X = self.vectorizer.fit_transform(xseq_array)
        y = np.asarray(yseq)
        self.mnb.partial_fit(X, y, classes=unique_classes)

    def save(self, model_file: str) -> None:
        """
        Save the model to a file.
        """
        # Dump both the model and the vectorizer
        dump(self.mnb, model_file)
        dump(self.vectorizer, self._get_vec_filename(model_file))

    @staticmethod
    def _get_vec_filename(model_file: str) -> str:
        """
        Get the vectorizer file name based on the model file name.
        """
        name, ext = os.path.splitext(model_file)
        return name + "_vectorizer" + ext

    @staticmethod
    def _to_dict(xseq: List[str]) -> Dict[str, str]:
        res = {}
        for feat in xseq:
            if "=" in feat:
                key, value = feat.split("=", 1)
                res[key] = value
            elif feat == "BoS":
                res["formPrev"] = "BoS"
                res["suf3Prev"] = "BoS"
            elif feat == "EoS":
                res["formNext"] = "EoS"
                res["suf3Next"] = "EoS"
        return res
