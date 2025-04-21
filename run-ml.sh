#! /bin/bash
set -e
set -u
set -o pipefail

# Check if the flag "--features" is passed
FEATURES=false
for arg in "$@"; do
  if [ "$arg" == "--features" ]; then
    FEATURES=true
  fi
done

if [ ! -d models ]; then
    mkdir models
fi

if [ ! -f data/train.feat ] || [ $FEATURES == true ]; then
    echo "Extracting training features..."
    python3 ml/extract-features.py data/train/ > data/train.feat
else
    echo "Training features already extracted. Use --features to re-extract."
fi
if [ ! -f data/devel.feat ] || [ $FEATURES == true ]; then
    echo "Extracting devel features..."
    python3 ml/extract-features.py data/devel/ > data/devel.feat
else
    echo "Devel features already extracted. Use --features to re-extract."
fi


echo "Training CRF model..."
python3 ml/train.py crf models/model.crf < data/train.feat

echo "Running CRF model..."
python3 ml/predict.py crf models/model.crf < data/devel.feat > data/devel-CRF.out

echo "Evaluating CRF results..."
python3 ml/evaluator.py NER data/devel data/devel-CRF.out > data/devel-CRF.stats

echo "Training Naive Bayes model..."
python3 ml/train.py mnb models/multinomial_nb.joblib < data/train.feat

echo "Running Naive Bayes model..."
python3 ml/predict.py mnb models/multinomial_nb.joblib < data/devel.feat > data/devel-NB.out

echo "Evaluating Naive Bayes results..."
python3 ml/evaluator.py NER data/devel data/devel-NB.out > data/devel-NB.stats

