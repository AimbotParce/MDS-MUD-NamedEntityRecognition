import re
import string
from typing import Optional

import numpy as np
from dataset import *
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Codemaps:
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data: Dataset | str, maxlen: Optional[int] = None, suflen: Optional[int] = None):

        self.word_index: dict[str, int]
        self.suf_index: dict[str, int]
        self.label_index: dict[str, int]
        self.maxlen: int
        self.suflen: int

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

        for sentence_tokens in data.sentences():
            for tagged_token in sentence_tokens:
                words.add(tagged_token["form"])
                suffixes.add(tagged_token["lc_form"][-self.suflen :])
                labels.add(tagged_token["tag"])

        self.word_index = {w: i + 2 for i, w in enumerate(list(words))}
        self.word_index["PAD"] = 0  # Padding
        self.word_index["UNK"] = 1  # Unknown words

        self.suf_index = {s: i + 2 for i, s in enumerate(list(suffixes))}
        self.suf_index["PAD"] = 0  # Padding
        self.suf_index["UNK"] = 1  # Unknown suffixes

        self.label_index = {t: i + 1 for i, t in enumerate(list(labels))}
        self.label_index["PAD"] = 0  # Padding

    ## --------- load indexes -----------
    def __load(self, name: str):
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}

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

    ## ---------- Save model and indexs ---------------
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

    def _encode_label(self, label: str) -> int:
        """
        Encode a label into its corresponding index.
        If the label is not found, return the index for "UNK".
        """
        return self.label_index.get(label, self.label_index["UNK"])

    ## --------- encode X from given data -----------
    def encode_words(self, data: Dataset):
        # encode and pad sentence words
        Xw = [[self._encode_word(token["form"]) for token in sentence] for sentence in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index["PAD"])
        # encode and pad suffixes
        Xs = [
            [self._encode_suffix(token["lc_form"][-self.suflen :]) for token in sentence]
            for sentence in data.sentences()
        ]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index["PAD"])
        # return encoded sequences
        return [Xw, Xs]

    ## --------- encode Y from given data -----------
    def encode_labels(self, data):
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
