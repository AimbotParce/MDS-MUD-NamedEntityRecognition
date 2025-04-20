#!/usr/bin/env python3

import sys

from joblib import load
from src.feature_space import SentenceYielder
from src.predictions import print_predictions


def fix_format(token):
    if "BoS" in token:
        token = token.replace("BoS", "formPrev=BoS\tsuf3Prev=BoS")
    if "EoS" in token:
        token = token.replace("EoS", "formNext=EoS\tsuf3Next=EoS")
    return token


def prepare_instances(xseq):
    features = []
    for token in xseq:
        token = fix_format("\t".join(token)).split("\t")
        token_dict = {feat.split("=")[0]: feat.split("=")[1] for feat in token[1:]}
        features.append(token_dict)
    return features


if __name__ == "__main__":

    # load leaned model and DictVectorizer
    model = load(sys.argv[1])
    v = load(sys.argv[2])

    # Read training instances from STDIN, and send them to trainer.
    for xseq, toks in SentenceYielder(sys.stdin)[5:, :4]:
        if len(xseq) == 0:
            continue
        xseq = prepare_instances(xseq)
        vectors = v.transform(xseq)
        predictions = model.predict(vectors)
        print_predictions(predictions, toks)
