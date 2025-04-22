#! /bin/bash
set -e
set -u
set -o pipefail

# train NN
echo "Training NN"
python3 nn/train.py data/train data/devel models/mymodel

# run model on devel data and compute performance
echo "Predicting and evaluatig"
python3 nn/predict.py models/mymodel data/devel data/devel-nn.out | tee data/devel-nn.stats
