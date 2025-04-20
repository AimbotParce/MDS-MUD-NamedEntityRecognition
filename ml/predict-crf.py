#!/usr/bin/env python3

import sys

from src.feature_space import SentenceYielder
from src.models import Model
from src.predictions import print_predictions


def main(model_file: str):
    model = Model(model_file)  # Load leaned model
    for xseq, toks in SentenceYielder(sys.stdin)[5:, :4]:
        predictions = model.predict(xseq)
        print_predictions(predictions, toks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: predict.py modelfile")
        sys.exit(1)
    model_file = sys.argv[1]
    if not model_file:
        print("Error: No model file specified.")
        sys.exit(1)

    main(model_file)
