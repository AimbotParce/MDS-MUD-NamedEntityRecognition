from typing import List, Optional
import joblib
import numpy as np

from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from sklearn.metrics import make_scorer, f1_score

from . import ModelABC

class CustomModel(ModelABC):
    aliases = "custom"  # Do not edit this

    def __init__(self, datafile: Optional[str] = None):
        self.model = None
        self.datafile = datafile
        if datafile is not None:
            self.model = joblib.load(datafile)

    def predict(self, xseq: List[List[str]]) -> List[str]:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        X = np.array([[float(feat) for feat in x] for x in xseq])
        preds = self.model.predict(X)
        return preds.tolist()

    def fit(self, xseq: List[List[str]], yseq: List[str], classes: Optional[List[str]] = None, **kwargs) -> None:
        X = np.array([[float(feat) for feat in x] for x in xseq])
        y = np.array(yseq)

        # Define sklearn macro F1 scorer
        f1_macro_scorer = make_scorer(f1_score, average="macro")

        self.model = AutoSklearn2Classifier(
            time_left_for_this_task=1800,  # 30 minutes total
            per_run_time_limit=180,        # 3 minutes per model fit ATMOST
            #n_jobs=-1,                    # USE ALL CORES
            n_jobs=10,                     # USE 10 CORES
            seed=42,
            scoring_functions=[f1_macro_scorer],  # can pass list of metrics here
            metric=f1_macro_scorer,
        )

        self.model.fit(X, y)

    def save(self, model_file: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        joblib.dump(self.model, model_file)
