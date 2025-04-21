#!/usr/bin/env python3

import sys

from src.feature_space import SentenceYielder
from src.models import load_model
from src.predictions import print_predictions


def main(model_type: str, model_file: str):
    model = load_model(model_file, model_type=model_type)  # Load leaned model
    for xseq, toks in SentenceYielder(sys.stdin)[5:, :4]:
        predictions = model.predict(xseq)
        print_predictions(predictions, toks)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: predict.py model_type model_file")
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
