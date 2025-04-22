import sys
from itertools import batched
from tempfile import mktemp
from typing import List, Optional

import pycrfsuite
import optuna

import logging
# use this to control / suppress Optuna info/debug logs
optuna.logging.set_verbosity(logging.WARNING)  


from . import ModelABC


class CustomModel(ModelABC):
    aliases = "custom"  # Do not edit this

    def __init__(self, datafile: Optional[str] = None):
        """
        Initialize the custom model. If datafile is provided, load the model from the file.
        """
        if datafile is not None:
            self.weights = datafile
        else:
            self.weights = mktemp(suffix=".crf")

    def predict(self, xseq: List[List[str]]) -> List[str]:
        """
        Predict the labels for the given sequences.
        """
        tagger = pycrfsuite.Tagger()
        tagger.open(self.weights)
        return tagger.tag(xseq)

    def fit(self, xseq: List[List[str]], yseq: List[str], classes: Optional[List[str]] = None, **kwargs) -> None:
        """
        Fit the model with hyperparameter optimization using Optuna.
        Note: expects validation data as kwargs: x_val, y_val for evaluation.
        """
        def objective(trial):
            minfreq = trial.suggest_int("feature_minfreq", 1, 5)
            c2 = trial.suggest_loguniform("c2", 1e-4, 10)
            max_iterations = trial.suggest_int("max_iterations", 10, 1000)

            trainer = pycrfsuite.Trainer()
            for x, y in zip(batched(xseq, 100), batched(yseq, 100)):
                trainer.append(x, y, 0)

            trainer.select("l2sgd", "crf1d")
            trainer.set("feature.minfreq", minfreq)
            trainer.set("c2", c2)
            trainer.set("max_iterations", max_iterations)

            trainer.train(self.weights)

            tagger = pycrfsuite.Tagger()
            tagger.open(self.weights)

            x_val = kwargs.get("x_val")
            y_val = kwargs.get("y_val")

            if x_val is None or y_val is None:
                # No validation data, skip evaluation by returning a default value
                return 0.0

            y_pred = [tagger.tag(x) for x in x_val]
            accuracy = self.evaluate_accuracy(y_pred, y_val)
            return -accuracy  # minimize negative accuracy

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print("Best hyperparameters:", study.best_params)
        print("Best accuracy:", -study.best_value)
        best_params = study.best_params

        # Re-train on full training data with best hyperparameters
        trainer = pycrfsuite.Trainer()
        for x, y in zip(batched(xseq, 100), batched(yseq, 100)):
            trainer.append(x, y, 0)

        trainer.select("l2sgd", "crf1d")
        trainer.set("feature.minfreq", best_params["feature_minfreq"])
        trainer.set("c2", best_params["c2"])
        trainer.set("max_iterations", best_params["max_iterations"])

        trainer.train(self.weights)

    def evaluate_accuracy(self, y_pred, y_val) -> float:
        correct = sum(
            pred_tag == true_tag
            for pred_seq, true_seq in zip(y_pred, y_val)
            for pred_tag, true_tag in zip(pred_seq, true_seq)
        )
        total = sum(len(seq) for seq in y_val)
        return correct / total if total > 0 else 0.0

    def save(self, model_file: str) -> None:
        """
        Save the model to a file.
        """
        with open(model_file, "wb") as f_out, open(self.weights, "rb") as f_in:
            f_out.write(f_in.read())
