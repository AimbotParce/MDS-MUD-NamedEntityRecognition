#! /bin/bash
set -e
set -u
set -o pipefail

# Check if the flag "--force" is passed
FORCE=false
for arg in "$@"; do
  if [ "$arg" == "--force" ]; then
    FORCE=true
  fi
done

if [ ! -f data/train.feat ] || [ $FORCE -eq true ]; then
    echo "Extracting training features..."
    python3 ml/extract-features.py data/train/ > data/train.feat
fi
if [ ! -f data/devel.feat ] || [ $FORCE -eq true ]; then
    echo "Extracting devel features..."
    python3 ml/extract-features.py data/devel/ > data/devel.feat
fi

echo "Training CRF model..."
python3 ml/train-crf.py models/model.crf < data/train.feat

echo "Running CRF model..."
python3 ml/predict.py models/model.crf < data/devel.feat > data/devel-CRF.out

echo "Evaluating CRF results..."
python3 ml/evaluator.py NER data/devel data/devel-CRF.out > data/devel-CRF.stats


#Extract Classification Features
cat data/train.feat | cut -f5- | grep -v ^$ > data/train.clf.feat


# train Naive Bayes model
echo "Training Naive Bayes model..."
python3 ml/train-sklearn.py models/model.joblib models/vectorizer.joblib < data/train.clf.feat
# run Naive Bayes model
echo "Running Naive Bayes model..."
python3 ml/predict-sklearn.py models/model.joblib models/vectorizer.joblib < data/devel.feat > data/devel-NB.out
# evaluate Naive Bayes results 
echo "Evaluating Naive Bayes results..."
python3 ml/evaluator.py NER data/devel data/devel-NB.out > data/devel-NB.stats

# remove auxiliary files.
rm data/train.clf.feat
