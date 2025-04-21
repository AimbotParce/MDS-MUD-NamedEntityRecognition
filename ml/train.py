#!/usr/bin/env python3
import sys
from itertools import chain

from src.feature_space import SentenceYielder
from src.models import load_model


def main(model_type: str, model_file: str):
    """
    Read training instances from STDIN, and train a model.
    The model is saved to the specified file.
    Args:
        model_type (str): The type of the model to be trained.
        model_file (str): The file where the trained model will be saved.
    """
    model = load_model(model_type=model_type)
    x_train, y_train = zip(*SentenceYielder(sys.stdin)[5:, 4])
    x_train = list(chain.from_iterable(x_train))
    y_train = list(chain.from_iterable(y_train))
    model.fit(x_train, y_train)
    model.save(model_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: train-crf.py model_type model_file")
        sys.exit(1)
    model_type = sys.argv[1]
    model_file = sys.argv[2]
    if not model_type:
        print("Error: No model type specified.")
        sys.exit(1)
    if not model_file:
        print("Error: No model file specified.")
        sys.exit(1)

    main(model_type, model_file)
