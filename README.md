# Mining Unstructured Data: Named Entity Recognition

The aim of this project is to set up a system for discovering named entities in text. Specifically, the focus lies on extracting the names of drugs, drug groups, drug brands, etc from a collection of sentences.

## Methodology

We adopted the B-I-O (Beginning, Inside, Outside) tagging scheme to annotate named entities in our text data, extending it with class-specific labels tailored to our domain. Each token in a sentence is labeled according to whether it begins (B-) or continues (I-) an entity, or falls outside any entity (O). The entity classes include `drug_n` (specific drug names), `drug` (general drug mentions), `group` (drug groups), and `brand` (commercial drug brands). For example, a phrase like “acetaminophen tablet” might be labeled as `B-drug_n I-drug_n`.

The workflow used is as follows:
1. Features are extracted using a feature extraction strategy, which can be changed for each of the experiments, for both the training and validation datasets. Feature lists are stored in two different files.
2. A training script loads the training features file and trains a model to predict the labels. The model is saved to disk, alongside any other trained objects (code maps, for example).
3. A predict script loads the model and the test features file and predicts the label for each of the words. Resulting tags are stored in a file.
4. An evaluator script loads the predicted tags, and the test features file, compares the two and spits out a table with the statistics.

## Structure of this repo

* For simplicity, the full dataset used can be found in the [`data/`](data/) folder, which is split into [`data/train/`](data/train/) and [`data/devel/`](data/devel/) for training and validation respectively.
* Each document in the dataset comes from a Drug-Drug Interaction dataset, and contains multiple sentences, alongside the "windows" within them that refer to a drug.
* The folders [`ml/`](ml/) and [`nn/`](nn/) are independent of one another, and each implement a different approach towards solving the problem. The first one extracts key-value sets of features for each of the words and trains classical machine learning models, whilst the second one encodes the sentences in tensors, and trains a neural network.
* The `main` branch is reserved for the baseline code, provided by the professors, and this branch is protected to disallow any modifications.
* Each experiment is devised and implemented in branches, and once an interesting* result is reached, a new pull request is opened.
* A **workflow** will kick off, running all the preprocessing and training steps and adding a comment to the PR with the statistics of the models.

> [!NOTE]
> An **interesting experiment** is not necessarily one that yields good results, but rather any experiment that tries something new, regardless of the results it gives.
