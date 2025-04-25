import os
from typing import List, Optional, Dict

import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

import lightgbm as lgb
from catboost import CatBoostClassifier

from . import ModelABC


class CustomModel(ModelABC):
    aliases = "custom"

    def __init__(self, datafile: Optional[str] = None):
        self.vectorizer = DictVectorizer()
        self.label_encoder = LabelEncoder()
        self._is_fitted = False

        # Base models
        self.svm_model = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
        self.lgbm_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
        self.catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, random_seed=42)

        # Meta-model for stacking
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)

        self.stacking_model = None

        if datafile is not None and os.path.exists(datafile):
            self._load(datafile)

    def fit(self, xseq: List[List[str]], yseq: List[str], **kwargs) -> None:
        # Convert features to dict then vectorize
        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.fit_transform(x_dicts)
        y = np.array(yseq)
        y_enc = self.label_encoder.fit_transform(y)

        # Setup stacking classifier
        self.stacking_model = StackingClassifier(
            estimators=[
                ("svm", self.svm_model),
                ("lgbm", self.lgbm_model),
                ("catboost", self.catboost_model),
            ],
            final_estimator=self.meta_model,
            cv=5,               # 5-fold cross-validation for out-of-fold predictions
            stack_method="predict_proba",  # Use probabilities for meta-model input
            n_jobs=-1,
            passthrough=True,  # If True, original features also fed to meta-model
        )

        self.stacking_model.fit(X, y_enc)
        self._is_fitted = True

    def predict(self, xseq: List[List[str]]) -> List[str]:
        if not self._is_fitted:
            raise RuntimeError("Model is not trained yet.")

        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.transform(x_dicts)

        preds = self.stacking_model.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, xseq: List[List[str]]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model is not trained yet.")

        x_dicts = [self._to_dict(x) for x in xseq]
        X = self.vectorizer.transform(x_dicts)
        return self.stacking_model.predict_proba(X)

    def save(self, model_file: str) -> None:
        joblib.dump(self.stacking_model, model_file)
        joblib.dump(self.vectorizer, self._get_vec_filename(model_file))
        joblib.dump(self.label_encoder, self._get_label_encoder_filename(model_file))

    def _load(self, model_file: str) -> None:
        self.stacking_model = joblib.load(model_file)
        self.vectorizer = joblib.load(self._get_vec_filename(model_file))
        self.label_encoder = joblib.load(self._get_label_encoder_filename(model_file))
        self._is_fitted = True

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

    @staticmethod
    def _get_vec_filename(model_file: str) -> str:
        name, ext = os.path.splitext(model_file)
        return name + "_vectorizer" + ext

    @staticmethod
    def _get_label_encoder_filename(model_file: str) -> str:
        name, ext = os.path.splitext(model_file)
        return name + "_label_encoder" + ext
