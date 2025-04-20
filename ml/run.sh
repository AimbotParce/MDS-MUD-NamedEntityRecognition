#! /bin/bash

BASEDIR=../../..

# convert datasets to feature vectors
echo "Extracting features..."
python3 extract-features.py $BASEDIR/data/train/ > train.feat
python3 extract-features.py $BASEDIR/data/devel/ > devel.feat

# train CRF model
echo "Training CRF model..."
python3 train-crf.py model.crf < train.feat
# run CRF model
echo "Running CRF model..."
python3 predict.py model.crf < devel.feat > devel-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python3 evaluator.py NER $BASEDIR/data/devel devel-CRF.out > devel-CRF.stats


#Extract Classification Features
cat train.feat | cut -f5- | grep -v ^$ > train.clf.feat


# train Naive Bayes model
echo "Training Naive Bayes model..."
python3 train-sklearn.py model.joblib vectorizer.joblib < train.clf.feat
# run Naive Bayes model
echo "Running Naive Bayes model..."
python3 predict-sklearn.py model.joblib vectorizer.joblib < devel.feat > devel-NB.out
# evaluate Naive Bayes results 
echo "Evaluating Naive Bayes results..."
python3 evaluator.py NER $BASEDIR/data/devel devel-NB.out > devel-NB.stats

# remove auxiliary files.
rm train.clf.feat
