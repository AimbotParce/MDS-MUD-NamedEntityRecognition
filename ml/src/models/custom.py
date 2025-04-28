import os
import warnings
from typing import List, Optional, Dict

import numpy as np
import lightgbm as lgb
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
        preds_prob = self.model.predict(X)
        preds_int = np.argmax(preds_prob, axis=1)  # pick class with highest probability
        return self.label_encoder.inverse_transform(preds_int)

    def fit(self, xseq: List[List[str]], yseq: List[str], **kwargs) -> None:
        # Silence warnings globally during training
        warnings.filterwarnings("ignore")

        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.fit_transform(x_dicts)
        y = np.array(yseq)
        y_enc = self.label_encoder.fit_transform(y)

        train_data = lgb.Dataset(X, label=y_enc)

        def objective(trial: optuna.Trial):
            param = {
                "objective": "multiclass",
                "num_class": len(self.label_encoder.classes_),
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_threads": -1,
                # Hyperparameters to tune
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            }
            cv_results = lgb.cv(
                param,
                train_data,
                nfold=3,
                stratified=True,
                num_boost_round=100,
                seed=42,
            )
            # Return best score (lowest multi_logloss)
            return min(cv_results["valid multi_logloss-mean"])

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(
            {
                "objective": "multiclass",
                "num_class": len(self.label_encoder.classes_),
                "metric": "multi_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "num_threads": -1,
            }
        )

        num_boost_round = kwargs.get("num_boost_round", 100)
        self.model = lgb.train(best_params, train_data, num_boost_round=num_boost_round)
        self._is_fitted = True

        # Restore warnings
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
