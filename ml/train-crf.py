#!/usr/bin/env python3

import sys

import pycrfsuite
from src.feature_space import SentenceYielder


def main(model_file: str):
    """
    Read training instances from STDIN, and train a CRF model.
    The model is saved to the specified file.
    Args:
        model_file (str): The file where the trained model will be saved.
    """
    # Create a Trainer object.
    trainer = pycrfsuite.Trainer()

    # Read training instances from STDIN, and append them to the trainer.
    for xseq, yseq in SentenceYielder(sys.stdin)[5:, 4]:
        trainer.append(xseq, yseq, 0)

    # Use L2-regularized SGD and 1st-order dyad features.
    trainer.select("l2sgd", "crf1d")

    # This demonstrates how to list parameters and obtain their values.
    trainer.set("feature.minfreq", 1)  # mininum frequecy of a feature to consider it
    trainer.set("c2", 0.1)  # coefficient for L2 regularization

    print("Training with following parameters: ")
    for name in trainer.params():
        print(name, trainer.get(name), trainer.help(name), file=sys.stderr)

    # Start training and dump model to modelfile
    trainer.train(model_file, -1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: train-crf.py modelfile")
        sys.exit(1)
    model_file = sys.argv[1]
    if not model_file:
        print("Error: No model file specified.")
        sys.exit(1)

    main(model_file)
