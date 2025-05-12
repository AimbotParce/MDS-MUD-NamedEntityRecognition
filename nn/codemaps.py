import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from dataset import *
from keras.preprocessing.sequence import pad_sequences
from numpy.typing import NDArray

nltk.download("averaged_perceptron_tagger_eng")
nltk.download("universal_tagset")
nltk.download("words")


def data_file(filename: str) -> Path:
    """
    Returns the path to the data file.
    """
    return (Path(__file__).parent.parent / "data" / filename).resolve()


class CustomSyllableTokenizer(nltk.tokenize.LegalitySyllableTokenizer):
    def tokenize(self, text: str) -> list[str]:
        # Custom tokenization logic
        return list(filter(lambda x: x.strip() != "", super().tokenize(text.replace("\n", " ").replace("\r", " "))))


class Codemaps:
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(
        self, data: Dataset | str, max_sentence_length: Optional[int] = None, max_word_length: Optional[int] = None
    ):
        self._syllable_tokenizer = CustomSyllableTokenizer(nltk.corpus.words.words())
        self.syllable_index: Dict[str, int]
        self.label_index: Dict[str, int]
        self.pos_tag_index: Dict[str, int]
        self.max_sentence_length: int
        self.max_word_length: int
        self.class_counts: Dict[str, int]

        if isinstance(data, Dataset) and max_sentence_length is not None and max_word_length is not None:
            self.__create_indexes(data, max_sentence_length, max_word_length)

        elif isinstance(data, str) and max_sentence_length is None and max_word_length is None:
            self.__load(data)

        else:
            print("codemaps: Invalid or missing parameters in constructor")
            exit(1)

    # --------- Create indexes from training data
    # Extract all words and labels in given sentences and
    # create indexes to encode them as numbers when needed
    def __create_indexes(self, data: Dataset, max_sentence_length: int, max_word_length: int):
        self.max_sentence_length = max_sentence_length
        self.max_word_length = max_word_length
        syllables = set[str]()
        labels = set[str]()
        pos_tags = set[str]()
        class_counts: defaultdict[str, int] = defaultdict(int)
        sentences_too_long: list[str] = list()
        words_too_long: list[str] = list()
        total_words = 0
        for sentence in data.sentences():
            if len(sentence) > self.max_sentence_length:
                sentences_too_long.append(
                    " ".join([token["lc_form"] for token in sentence]) + " (" + str(len(sentence)) + " words)"
                )
            for tagged_token in sentence:
                syls = self._syllable_tokenizer.tokenize(tagged_token["lc_form"])
                if len(syls) > self.max_word_length:
                    words_too_long.append(tagged_token["lc_form"] + " (" + str(len(syls)) + " syllables)")
                syllables.update(syls)
                labels.add(tagged_token["tag"])
                class_counts[tagged_token["tag"]] += 1
                total_words += 1

            sentence_pos_tags = nltk.tag.pos_tag([token["lc_form"] for token in sentence], tagset="universal")
            pos_tags.update([tag for _, tag in sentence_pos_tags])

            class_counts["PAD"] += max(self.max_sentence_length - len(sentence), 0)

        if len(sentences_too_long) > 0:
            print(
                "WARNING: The following %d sentences are too long, and will end up truncated:"
                % len(sentences_too_long),
                file=sys.stderr,
            )
            for sentence in sentences_too_long:
                print("  ", sentence, file=sys.stderr)
        if len(words_too_long) > 0:
            print(
                "WARNING: The following %d words instances out of the total %s are too long, and will end up truncated:"
                % (len(words_too_long), total_words),
                file=sys.stderr,
            )
            for word in words_too_long:
                print("  ", word, file=sys.stderr)

        self.class_counts = dict(class_counts)

        self.syllable_index = {w: i + 2 for i, w in enumerate(list(syllables))}
        self.syllable_index["PAD"] = 0  # Padding
        self.syllable_index["UNK"] = 1  # Unknown words

        self.label_index = {t: i + 2 for i, t in enumerate(list(labels))}
        self.label_index["PAD"] = 0  # Padding
        self.label_index["UNK"] = 1  # Unknown labels

        self.pos_tag_index = {t: i + 2 for i, t in enumerate(list(pos_tags))}
        self.pos_tag_index["PAD"] = 0  # Padding
        self.pos_tag_index["UNK"] = 1  # Unknown labels

    ## --------- load indexes -----------
    def __load(self, name: str):
        self.max_sentence_length = 0
        self.max_word_length = 0
        self.syllable_index = {}
        self.label_index = {}
        self.pos_tag_index = {}
        self.class_counts: Dict[str, int] = {}

        with open(name + ".idx") as f:
            for line in f.readlines():
                (t, k, i) = line.split()
                if t == "MAX_SENTENCE_LENGTH":
                    self.max_sentence_length = int(k)
                elif t == "MAX_WORD_LENGTH":
                    self.max_word_length = int(k)
                elif t == "SYLLABLE":
                    self.syllable_index[k] = int(i)
                elif t == "LABEL":
                    self.label_index[k] = int(i)
                elif t == "COUNT":
                    self.class_counts[k] = int(i)
                elif t == "POS":
                    self.pos_tag_index[k] = int(i)

    ## ---------- Save model and indexes ---------------
    def save(self, name: str):
        # save indexes
        with open(name + ".idx", "w") as f:
            print("MAX_SENTENCE_LENGTH", self.max_sentence_length, "-", file=f)
            print("MAX_WORD_LENGTH", self.max_word_length, "-", file=f)
            for key in self.label_index:
                print("LABEL", key, self.label_index[key], file=f)
            for key in self.syllable_index:
                print("SYLLABLE", key, self.syllable_index[key], file=f)
            for key in self.class_counts:
                print("COUNT", key, self.class_counts[key], file=f)
            for key in self.pos_tag_index:
                print("POS", key, self.pos_tag_index[key], file=f)

    def _encode_syllable(self, syllable: str) -> int:
        """
        Encode a word into its corresponding index.
        If the word is not found, return the index for "UNK".
        """
        return self.syllable_index.get(syllable, self.syllable_index["UNK"])

    def _encode_label(self, label: str) -> int:
        """
        Encode a label into its corresponding index.
        If the label is not found, return the index for "UNK".
        """
        return self.label_index.get(label, self.label_index["UNK"])

    def _encode_pos_tag(self, pos_tag: str) -> int:
        """
        Encode a POS tag into its corresponding index.
        If the POS tag is not found, return the index for "UNK".
        """
        return self.pos_tag_index.get(pos_tag, self.pos_tag_index["UNK"])

    ## --------- encode X from given data -----------
    def encode_words(self, data: Dataset) -> NDArray[np.int32]:
        # encode and pad sentence words

        X_word = [
            pad_sequences(
                maxlen=self.max_word_length,
                sequences=[
                    [self._encode_syllable(syl) for syl in self._syllable_tokenizer.tokenize(token["lc_form"])]
                    for token in sentence
                ],
                padding="post",
                value=self.syllable_index["PAD"],
            )
            for sentence in data.sentences()
        ]
        X_word = pad_sequences(
            maxlen=self.max_sentence_length,
            sequences=X_word,
            padding="post",
            value=[self.syllable_index["PAD"]] * self.max_word_length,
        )

        X_pos_tags = [
            [
                self._encode_pos_tag(tag)
                for tag in nltk.tag.pos_tag([token["lc_form"] for token in sentence], tagset="universal")
            ]
            for sentence in data.sentences()
        ]
        X_pos_tags = pad_sequences(
            maxlen=self.max_sentence_length,
            sequences=X_pos_tags,
            padding="post",
            value=self.pos_tag_index["PAD"],
        )

        # return encoded sequences
        return X_word, X_pos_tags

    ## --------- encode Y from given data -----------
    def encode_labels(self, data: Dataset) -> NDArray[np.int32]:
        # encode and pad sentence labels
        Y = [[self._encode_label(token["tag"]) for token in sentence] for sentence in data.sentences()]
        Y = pad_sequences(maxlen=self.max_sentence_length, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_syllables(self):
        return len(self.syllable_index)

    def get_n_pos_tags(self):
        return len(self.pos_tag_index)

    ## -------- get label index size ---------
    def get_n_labels(self):
        return len(self.label_index)

    ## -------- get index for given word ---------
    def word2idx(self, w: str):
        return self.syllable_index[w]

    ## -------- get index for given label --------
    def label2idx(self, l: str):
        return self.label_index[l]

    ## -------- get label name for given index --------
    def idx2label(self, i: int):
        for l in self.label_index:
            if self.label_index[l] == i:
                return l
        raise KeyError
