import os
import warnings
from typing import List, Optional, Dict

import numpy as np
import catboost as cb
import joblib
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

from . import ModelABC


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
        preds_prob = self.model.predict_proba(X)
        preds_int = np.argmax(preds_prob, axis=1)
        return self.label_encoder.inverse_transform(preds_int)

    def fit(self, xseq: List[List[str]], yseq: List[str], **kwargs) -> None:
        warnings.filterwarnings("ignore")

        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.fit_transform(x_dicts)
        y = np.array(yseq)
        y_enc = self.label_encoder.fit_transform(y)

        train_pool = cb.Pool(X, label=y_enc)

        def objective(trial: optuna.Trial):
            param = {
                "iterations": 1000,
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "random_strength": trial.suggest_float("random_strength", 0, 1),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "task_type": "CPU",
                "verbose": False,
                "loss_function": "MultiClass",
                "classes_count": len(self.label_encoder.classes_),
                "thread_count": os.cpu_count(),
                "early_stopping_rounds": 50,
            }

            cv_results = cb.cv(
                pool=train_pool,
                params=param,
                fold_count=3,
                stratified=True,
                shuffle=True,
                partition_random_seed=42,
                plot=False,
                verbose=False,
            )
            # cv_results is a dict with metric names as keys and lists of values per iteration as values
            # We want to minimize multi-class logloss
            best_loss = min(cv_results["test-MultiClass-mean"])
            return best_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(
            {
                "iterations": kwargs.get("iterations", 1000),
                "loss_function": "MultiClass",
                "verbose": False,
                "thread_count": -1,
                "classes_count": len(self.label_encoder.classes_),
            }
        )

        self.model = cb.CatBoostClassifier(**best_params)
        self.model.fit(train_pool, early_stopping_rounds=50, verbose=False)
        self._is_fitted = True

        warnings.resetwarnings()

    def save(self, model_file: str) -> None:
        self.model.save_model(model_file)
        joblib.dump(self.vectorizer, self._get_vec_filename(model_file))
        joblib.dump(self.label_encoder, self._get_label_encoder_filename(model_file))

    def _load(self, model_file: str) -> None:
        self.model = cb.CatBoostClassifier()
        self.model.load_model(model_file)
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
