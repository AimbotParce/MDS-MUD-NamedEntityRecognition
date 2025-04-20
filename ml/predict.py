#!/usr/bin/env python3

import sys

from src.feature_space import SentenceYielder
from src.models import Model


def main(model_file: str):
    model = Model(model_file)  # Load leaned model
    for xseq, toks in SentenceYielder(sys.stdin)[5:, :4]:
        predictions = model.predict(xseq)

        inside = False
        for k in range(0, len(predictions)):
            y = predictions[k]
            (sid, form, offS, offE) = toks[k]

            if y[0] == "B":
                entity_form = form
                entity_start = offS
                entity_end = offE
                entity_type = y[2:]
                inside = True
            elif y[0] == "I" and inside:
                entity_form += " " + form
                entity_end = offE
            elif y[0] == "O" and inside:
                print(sid, entity_start + "-" + entity_end, entity_form, entity_type, sep="|")
                inside = False

        if inside:
            print(sid, entity_start + "-" + entity_end, entity_form, entity_type, sep="|")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: predict.py modelfile")
        sys.exit(1)
    model_file = sys.argv[1]
    if not model_file:
        print("Error: No model file specified.")
        sys.exit(1)

    main(model_file)
