import os
import warnings
from typing import List, Optional, Dict

import numpy as np
import joblib
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

from . import ModelABC  # Your base class


class CustomModel(ModelABC):
    aliases = "custom"

    def __init__(self, datafile: Optional[str] = None):
        self.vectorizer = DictVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = None
        self._is_fitted = False

        if datafile is not None and os.path.exists(datafile):
            self._load(datafile)

    def predict(self, xseq: List[List[str]]) -> List[str]:
        if not self._is_fitted:
            raise RuntimeError("Model is not trained yet.")

        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.transform(x_dicts)
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def fit(self, xseq: List[List[str]], yseq: List[str], **kwargs) -> None:
        # Silence warnings globally during training
        warnings.filterwarnings("ignore")

        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.fit_transform(x_dicts)
        y = np.array(yseq)
        y_enc = self.label_encoder.fit_transform(y)

        def objective(trial: optuna.Trial):
            # Suggest hyperparameters to tune
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
            gamma = "scale"

            if kernel == "poly":
                degree = trial.suggest_int("degree", 2, 5)  # Tune degree only if kernel == poly
                model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True, random_state=42)
            else:
                degree = None  # Not used
                model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)

            # Use stratified 3-fold CV and measure accuracy
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        best_params = study.best_params
        kernel = best_params.pop("kernel")

        if kernel == "poly":
            degree = best_params.pop("degree")
            self.model = SVC(kernel=kernel, degree=degree, probability=True, random_state=42, **best_params)
        else:
            self.model = SVC(kernel=kernel, probability=True, random_state=42, **best_params)

        self.model.fit(X, y_enc)
        self._is_fitted = True

        warnings.resetwarnings()

    def save(self, model_file: str) -> None:
        joblib.dump(self.model, model_file)
        joblib.dump(self.vectorizer, self._get_vec_filename(model_file))
        joblib.dump(self.label_encoder, self._get_label_encoder_filename(model_file))

    def _load(self, model_file: str) -> None:
        self.model = joblib.load(model_file)
        self.vectorizer = joblib.load(self._get_vec_filename(model_file))
        self.label_encoder = joblib.load(self._get_label_encoder_filename(model_file))
        self._is_fitted = True

    @staticmethod
    def _get_vec_filename(model_file: str) -> str:
        name, ext = os.path.splitext(model_file)
        return name + "_vectorizer" + ext

    @staticmethod
    def _get_label_encoder_filename(model_file: str) -> str:
        name, ext = os.path.splitext(model_file)
        return name + "_label_encoder" + ext

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
