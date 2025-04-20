#! /bin/bash

BASEDIR=../../..
export PYTHONPATH=$BASEDIR/util

# train NN
echo "Training NN"
python3 train.py $BASEDIR/data/train $BASEDIR/data/devel mymodel

# run model on devel data and compute performance
echo "Predicting and evaluatig"
python3 predict.py mymodel $BASEDIR/data/devel devel.out | tee devel.stats
