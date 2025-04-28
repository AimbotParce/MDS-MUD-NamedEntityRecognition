#! /bin/bash
set -e
set -u
set -o pipefail

if [ ! -d models ]; then
    mkdir models
fi

# train NN
echo "Training NN"
# python3 nn/train.py data/train data/devel models/mymodel.h5

# run model on devel data and compute performance
echo "Predicting and evaluatig"
python3 nn/predict.py models/mymodel.h5 data/devel data/devel-nn.out | tee data/devel-nn.stats
# python3 nn/predict.py models/mymodel.h5 data/train data/train-nn.out | tee data/train-nn.stats
