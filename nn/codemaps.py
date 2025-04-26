import json
import re
import string
from pathlib import Path
from typing import Optional

import numpy as np
from dataset import *
from keras import KerasTensor
from numpy.typing import NDArray
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    def __init__(self, data: Dataset | str, maxlen: Optional[int] = None, suflen: Optional[int] = None):

        self.word_index: Dict[str, int]
        self.suf_index: Dict[str, int]
        self.label_index: Dict[str, int]
        self.maxlen: int
        self.suflen: int
        self.pos_tag_index: Dict[str, int]

        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexes(data, maxlen, suflen)

        elif isinstance(data, str) and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit(1)

    # --------- Create indexes from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexes(self, data: Dataset, maxlen: int, suflen: int):
        self.maxlen = maxlen
        self.suflen = suflen
        words = set[str]()
        lowercase_words = set[str]()
        suffixes = set[str]()
        labels = set[str]()
        pos_tags = set[str]()

        for sentence in data.sentences():
            sent_pos_tags = nltk.tag.pos_tag([token["lc_form"] for token in sentence], tagset="universal")
            pos_tags.update([pos_tag for _, pos_tag in sent_pos_tags])
            for tagged_token in sentence:
                words.add(tagged_token["lc_form"])
                suffixes.add(tagged_token["lc_form"][-self.suflen :])
                labels.add(tagged_token["tag"])

        self.word_index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word_index["PAD"] = 0  # Padding
        self.word_index["UNK"] = 1  # Unknown words

        self.suf_index = {s: i + 2 for i, s in enumerate(list(suffixes))}
        self.suf_index["PAD"] = 0  # Padding
        self.suf_index["UNK"] = 1  # Unknown suffixes

        self.label_index = {t: i + 2 for i, t in enumerate(list(labels))}
        self.label_index["PAD"] = 0  # Padding
        self.label_index["UNK"] = 1  # Unknown labels

        self.pos_tag_index = {t: i + 2 for i, t in enumerate(list(pos_tags))}
        self.pos_tag_index["PAD"] = 0  # Padding
        self.pos_tag_index["UNK"] = 1  # Unknown PoS tags

    ## --------- load indexes -----------
    def __load(self, name: str):
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}
        self.pos_tag_index = {}

        with open(name + ".idx") as f:
            for line in f.readlines():
                (t, k, i) = line.split()
                if t == "MAXLEN":
                    self.maxlen = int(k)
                elif t == "SUFLEN":
                    self.suflen = int(k)
                elif t == "WORD":
                    self.word_index[k] = int(i)
                elif t == "SUF":
                    self.suf_index[k] = int(i)
                elif t == "LABEL":
                    self.label_index[k] = int(i)
                elif t == "POS":
                    self.pos_tag_index[k] = int(i)

    ## ---------- Save model and indexes ---------------
    def save(self, name: str):
        # save indexes
        with open(name + ".idx", "w") as f:
            print("MAXLEN", self.maxlen, "-", file=f)
            print("SUFLEN", self.suflen, "-", file=f)
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)
            for key in self.word_index:
                print("WORD", key, self.word_index[key], file=f)
            for key in self.suf_index:
                print("SUF", key, self.suf_index[key], file=f)
            for key in self.pos_tag_index:
                print("POS", key, self.pos_tag_index[key], file=f)

    def _encode_word(self, word: str) -> int:
        """
        Encode a word into its corresponding index.
        If the word is not found, return the index for "UNK".
        """
        return self.word_index.get(word, self.word_index["UNK"])

    def _encode_suffix(self, suffix: str) -> int:
        """
        Encode a suffix into its corresponding index.
        If the suffix is not found, return the index for "UNK".
        """
        return self.suf_index.get(suffix, self.suf_index["UNK"])

    def _encode_pos_tag(self, pos_tag: str) -> int:
        """
        Encode a PoS tag into its corresponding index.
        If the PoS tag is not found, return the index for "UNK".
        """
        # Assuming PoS tags are already in the label_index
        return self.pos_tag_index.get(pos_tag, self.pos_tag_index["UNK"])

    def _encode_label(self, label: str) -> int:
        """
        Encode a label into its corresponding index.
        If the label is not found, return the index for "UNK".
        """
        return self.label_index.get(label, self.label_index["UNK"])

    ## --------- encode X from given data -----------
    def encode_words(
        self, data: Dataset
    ) -> Tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
        # encode and pad sentence words
        X_word = [[self._encode_word(token["lc_form"]) for token in sentence] for sentence in data.sentences()]
        X_word = pad_sequences(maxlen=self.maxlen, sequences=X_word, padding="post", value=self.word_index["PAD"])
        # encode and pad suffixes
        X_suffix = [
            [self._encode_suffix(token["lc_form"][-self.suflen :]) for token in sentence]
            for sentence in data.sentences()
        ]
        X_suffix = pad_sequences(maxlen=self.maxlen, sequences=X_suffix, padding="post", value=self.suf_index["PAD"])
        # Save Word Lengths
        X_length = [[len(token["lc_form"]) for token in sentence] for sentence in data.sentences()]
        X_length = pad_sequences(maxlen=self.maxlen, sequences=X_length, padding="post", value=0)
        # Add PoS tags
        X_pos = []
        for sentence in data.sentences():
            pos_tags = nltk.tag.pos_tag([token["lc_form"] for token in sentence], tagset="universal")
            X_pos.append([self._encode_pos_tag(pos_tag) for _, pos_tag in pos_tags])
        X_pos = pad_sequences(maxlen=self.maxlen, sequences=X_pos, padding="post", value=self.pos_tag_index["PAD"])

        # return encoded sequences
        return X_word, X_suffix, X_pos, X_length

    ## --------- encode Y from given data -----------
    def encode_labels(self, data: Dataset) -> NDArray[np.int32]:
        # encode and pad sentence labels
        Y = [[self._encode_label(token["tag"]) for token in sentence] for sentence in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self):
        return len(self.word_index)

    ## -------- get suf index size ---------
    def get_n_sufs(self):
        return len(self.suf_index)

    ## -------- get label index size ---------
    def get_n_labels(self):
        return len(self.label_index)

    def get_n_pos_tags(self):
        return len(self.pos_tag_index)

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
