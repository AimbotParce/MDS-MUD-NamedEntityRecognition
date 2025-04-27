from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from dataset import *
from keras.preprocessing.sequence import pad_sequences
from numpy.typing import NDArray

nltk.download("averaged_perceptron_tagger_eng")
nltk.download("universal_tagset")


def data_file(filename: str) -> Path:
    """
    Returns the path to the data file.
    """
    return (Path(__file__).parent.parent / "data" / filename).resolve()


class Codemaps:
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data: Dataset | str, maxlen: Optional[int] = None):
        self.word_index: Dict[str, int]
        self.label_index: Dict[str, int]
        self.maxlen: int
        self.class_counts: Dict[str, int]

        if isinstance(data, Dataset) and maxlen is not None:
            self.__create_indexes(data, maxlen)

        elif isinstance(data, str) and maxlen is None:
            self.__load(data)

        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit(1)

    # --------- Create indexes from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexes(self, data: Dataset, maxlen: int):
        self.maxlen = maxlen
        words = set[str]()
        labels = set[str]()
        class_counts: defaultdict[str, int] = defaultdict(int)

        for sentence in data.sentences():
            for tagged_token in sentence:
                words.add(tagged_token["lc_form"])
                labels.add(tagged_token["tag"])
                class_counts[tagged_token["tag"]] += 1

            class_counts["PAD"] += max(self.maxlen - len(sentence), 0)

        self.class_counts = dict(class_counts)

        self.word_index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word_index["PAD"] = 0  # Padding
        self.word_index["UNK"] = 1  # Unknown words

        self.label_index = {t: i + 2 for i, t in enumerate(list(labels))}
        self.label_index["PAD"] = 0  # Padding
        self.label_index["UNK"] = 1  # Unknown labels

    ## --------- load indexes -----------
    def __load(self, name: str):
        self.maxlen = 0
        self.word_index = {}
        self.label_index = {}
        self.class_counts: Dict[str, int] = {}

        with open(name + ".idx") as f:
            for line in f.readlines():
                (t, k, i) = line.split()
                if t == "MAXLEN":
                    self.maxlen = int(k)
                elif t == "WORD":
                    self.word_index[k] = int(i)
                elif t == "LABEL":
                    self.label_index[k] = int(i)
                elif t == "COUNT":
                    self.class_counts[k] = int(i)

    ## ---------- Save model and indexes ---------------
    def save(self, name: str):
        # save indexes
        with open(name + ".idx", "w") as f:
            print("MAXLEN", self.maxlen, "-", file=f)
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)
            for key in self.word_index:
                print("WORD", key, self.word_index[key], file=f)
            for key in self.class_counts:
                print("COUNT", key, self.class_counts[key], file=f)

    def _encode_word(self, word: str) -> int:
        """
        Encode a word into its corresponding index.
        If the word is not found, return the index for "UNK".
        """
        return self.word_index.get(word, self.word_index["UNK"])

    def _encode_label(self, label: str) -> int:
        """
        Encode a label into its corresponding index.
        If the label is not found, return the index for "UNK".
        """
        return self.label_index.get(label, self.label_index["UNK"])

    ## --------- encode X from given data -----------
    def encode_words(self, data: Dataset) -> NDArray[np.int32]:
        # encode and pad sentence words
        X_word = [[self._encode_word(token["lc_form"]) for token in sentence] for sentence in data.sentences()]
        X_word = pad_sequences(maxlen=self.maxlen, sequences=X_word, padding="post", value=self.word_index["PAD"])
        # return encoded sequences
        return X_word

    ## --------- encode Y from given data -----------
    def encode_labels(self, data: Dataset) -> NDArray[np.int32]:
        # encode and pad sentence labels
        Y = [[self._encode_label(token["tag"]) for token in sentence] for sentence in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self):
        return len(self.word_index)

    ## -------- get label index size ---------
    def get_n_labels(self):
        return len(self.label_index)

    ## -------- get index for given word ---------
    def word2idx(self, w: str):
        return self.word_index[w]

    ## -------- get index for given label --------
    def label2idx(self, l: str):
        return self.label_index[l]

    ## -------- get label name for given index --------
    def idx2label(self, i: int):
        for l in self.label_index:
            if self.label_index[l] == i:
                return l
        raise KeyError
