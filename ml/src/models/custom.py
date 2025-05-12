import os
import warnings
from typing import List, Optional, Dict

import numpy as np
import xgboost as xgb
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
        dmatrix = xgb.DMatrix(X)
        preds_prob = self.model.predict(dmatrix)
        # preds_prob shape: (num_samples, num_classes)
        preds_int = np.argmax(preds_prob, axis=1)
        return self.label_encoder.inverse_transform(preds_int)

    def fit(self, xseq: List[List[str]], yseq: List[str], **kwargs) -> None:
        warnings.filterwarnings("ignore")

        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.fit_transform(x_dicts)
        y = np.array(yseq)
        y_enc = self.label_encoder.fit_transform(y)

        dtrain = xgb.DMatrix(X, label=y_enc)
        num_class = len(self.label_encoder.classes_)

        def objective(trial: optuna.Trial):
            param = {
                "objective": "multi:softprob",
                "num_class": num_class,
                "eval_metric": "mlogloss",
                "verbosity": 0,
                "tree_method": "hist",  # fast histogram based method
                "booster": "gbtree",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "lambda": trial.suggest_float("lambda", 0.0, 5.0),
                "alpha": trial.suggest_float("alpha", 0.0, 5.0),
                "nthread": -1,
            }

            cv_results = xgb.cv(
                param,
                dtrain,
                num_boost_round=100,
                nfold=3,
                stratified=True,
                seed=42,
                early_stopping_rounds=10,
                metrics=("mlogloss"),
                as_pandas=True,
                verbose_eval=False,
            )
            return cv_results["test-mlogloss-mean"].min()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100, show_progress_bar=False)

        best_params = study.best_params
        best_params.update(
            {
                "objective": "multi:softprob",
                "num_class": num_class,
                "eval_metric": "mlogloss",
                "verbosity": 0,
                "booster": "gbtree",
                "tree_method": "hist",
                "nthread": -1,
            }
        )

        num_boost_round = kwargs.get("num_boost_round", 100)
        self.model = xgb.train(best_params, dtrain, num_boost_round=num_boost_round)
        self._is_fitted = True

        warnings.resetwarnings()

    def save(self, model_file: str) -> None:
        self.model.save_model(model_file)
        joblib.dump(self.vectorizer, self._get_vec_filename(model_file))
        joblib.dump(self.label_encoder, self._get_label_encoder_filename(model_file))

    def _load(self, model_file: str) -> None:
        self.model = xgb.Booster()
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
