#! /usr/bin/python3
"""
Tokenize sentence, returning tokens and span offsets
"""
import os
import sys
from collections import defaultdict
from os import listdir
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypeAlias
from xml.dom.minidom import parse

import nltk
import nltk.tag
import numpy as np
from nltk.tokenize import word_tokenize

Token: TypeAlias = Tuple[str, int, int]
EntitySpan: TypeAlias = Tuple[int, int, str]

nltk.download("averaged_perceptron_tagger_eng")
nltk.download("universal_tagset")


def read_list(filename: str) -> List[str]:
    """
    Read a list of strings from a file, stripping whitespace and newlines.
    Args:
        filename (str): The name of the file to read.
    Returns:
        List[str]: A list of strings read from the file.
    """
    with open(filename, "r") as f:
        return list(line.strip() for line in f.readlines())


def data_path(filename: str) -> Path:
    """
    Get the absolute path of a file in the data directory.
    Args:
        filename (str): The name of the file.
    Returns:
        file_path (Path): The absolute path of the file.
    """
    return (Path(__file__).parent.parent / "data" / filename).resolve()


drug_suffixes = read_list(data_path("med-suffixes.txt"))  # Read drug suffixes from a file
drug_form_words = read_list(data_path("med-form-words.txt"))  # Read drug form words from a file


def tokenize(txt: str):
    """
    Tokenize a text string, returning a list of tokens and their
    start and end positions in the original string.

    Args:
        txt (str): The text string to tokenize.
    Returns:
        List[Token]: A list of tuples, each containing a token and its
                     start and end positions in the original string.
    """

    offset = 0
    tks: List[Token] = []
    for t in word_tokenize(txt):  # word_tokenize splits words, taking into account punctuations, numbers, etc.
        # Keep track of the position where each token should appear, and
        # store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)
    return tks  # tks is a list of triples (word,start,end)


def get_tag(token: Token, spans: List[EntitySpan]):
    """
    Determine the tag for a given token based on its position and the provided spans.
    Args:
        token (Token): A tuple containing the token and its start and end positions.
        spans (List[EntitySpan]): A list of entity spans, each defined by a start, end, and type.
    Returns:
        str: The tag for the token, which can be "B-", "I-", or "O".
    """
    (form, start, end) = token
    for spanS, spanE, spanT in spans:
        if start == spanS and end <= spanE:
            return "B-" + spanT
        elif start >= spanS and end <= spanE:
            return "I-" + spanT
    return "O"


def extract_features(tokens: List[Token]):
    """
    Extract a list of features for each token given the tokenized sentence.
    Args:
        tokens (List[Token]): A list of tuples, each containing a token and its
                              start and end positions in the original string.
    Returns:
        List[List[str]]: A list of lists, where each inner list contains
                         features for the corresponding token.
    """

    # for each token, generate list of features and add it to the result
    result: List[List[str]] = []
    pos_tags = nltk.tag.pos_tag([t[0] for t in tokens], tagset="universal")  # Get POS tags for the tokens
    assert len(tokens) == len(pos_tags), "Mismatch between tokens and POS tags length"
    for k, (word, pos_tag) in enumerate(pos_tags):
        features: Dict[str, str] = {}

        features["form"] = word  # Token form
        features["form-lower"] = word.lower()  # Lowercase form of the token
        features["is-capitalized"] = str(word[0].isupper())  # Is the first letter capitalized?
        features["is-uppercase"] = str(word.isupper())  # Is the token all uppercase?
        features["has-digit"] = str(any(c.isdigit() for c in word))  # Does the token contain a digit?
        features["has-punct"] = str(any(c in ".,;:!?" for c in word))  # Does the token contain punctuation?
        features["has-hyphen"] = str("-" in word)  # Does the token contain a hyphen?
        features["length"] = str(len(word))  # Length of the token
        features["is-long"] = str(len(word) > 5)  # Is the token long?
        features["pos-tag"] = pos_tag  # POS tag of the token
        features["is-bos"] = str(k == 0)  # Is the token at the beginning of the sentence?
        features["is-eos"] = str(k == len(tokens) - 1)  # Is the token at the end of the sentence?

        for suffix in drug_suffixes:
            if word.lower().endswith(suffix):
                features["has-med-suffix"] = "True"
                features["med-suffix"] = suffix
                break
        else:
            features["has-med-suffix"] = "False"
            features["med-suffix"] = "[NA]"
        for form_word in drug_form_words:
            if word.lower() == form_word:
                features["is-med-form-word"] = "True"
                break
        else:
            features["is-med-form-word"] = "False"

        # Context window features
        context_size = 5
        context_info = defaultdict(lambda: np.zeros(context_size, dtype=np.bool))
        for j, i in enumerate(range(-(context_size // 2), context_size - (context_size // 2))):
            if k + i < 0 or k + i >= len(tokens):
                # If this is outside the sentence
                continue
            context_word = tokens[k + i][0]
            if i != 0:
                context_word_features: Dict[str, str] = {}
                context_word_features["form-lower"] = context_word.lower()
                context_word_features["length"] = str(len(context_word))
                for key, val in context_word_features.items():
                    # Adds features like "ctx-l2-length=12", meaning "length of word two to the left is 12"
                    ctx_i = f"r{i}" if i > 0 else f"l{-i}"
                    features[f"ctx-{ctx_i}-{key}"] = val

            if context_word[0].isupper():
                context_info["is-capitalized"][j] = True
            if context_word.isupper():
                context_info["is-uppercase"][j] = True
            if any(c.isdigit() for c in context_word):
                context_info["has-digit"][j] = True
            if "-" in context_word:
                context_info["has-hyphen"][j] = True

        for key, vals in context_info.items():
            features[f"ctx-{key}"] = "".join("1" if v else "0" for v in vals)

        result.append(list(f"{key}={val}" for key, val in features.items()))

    return result


def main(datadir: str):
    """
    Main function to extract features from XML files in the specified directory.
    Args:
        datadir (str): The directory containing the XML files.
    """
    for f in listdir(datadir):  # List all files in the directory
        tree = parse(os.path.join(datadir, f))  # Parse XML file, obtaining a DOM tree
        sentence_elements = tree.getElementsByTagName("sentence")  # Get all sentences in the file
        for sentence in sentence_elements:
            sentence_id = sentence.attributes["id"].value  # Get sentence id
            spans: List[EntitySpan] = []
            sentence_content = sentence.attributes["text"].value  # Get sentence text
            entity_elements = sentence.getElementsByTagName("entity")  # Get pre-annotated entities
            for entity in entity_elements:
                # For discontinuous entities, we only get the first span
                # (will not work, but there are few of them)
                (start, end) = entity.attributes["charOffset"].value.split(";")[0].split("-")
                entity_type = entity.attributes["type"].value
                spans.append((int(start), int(end), entity_type))

            tokens = tokenize(sentence_content)  # Convert the sentence to a list of tokens
            features = extract_features(tokens)  # Extract sentence features

            # Print features in format expected by crfsuite trainer
            for i in range(0, len(tokens)):
                # See if the token is part of an entity
                tag = get_tag(tokens[i], spans)
                print(sentence_id, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep="\t")

            # blank line to separate sentences
            print()


if __name__ == "__main__":
    ## --------- MAIN PROGRAM -----------
    ## --
    ## -- Usage:  baseline-NER.py target-dir
    ## --
    ## -- Extracts Drug NE from all XML files in target-dir, and writes
    ## -- them in the output format requested by the evalution programs.
    ## --
    if len(sys.argv) != 2:
        print("Usage: extract-features.py target-dir")
        sys.exit(1)
    datadir = sys.argv[1]
    if not os.path.exists(datadir):
        print("Directory does not exist:", datadir)
        sys.exit(1)
    if not os.path.isdir(datadir):
        print("Not a directory:", datadir)
        sys.exit(1)
    if not os.access(datadir, os.R_OK):
        print("Cannot read directory:", datadir)
        sys.exit(1)

    nltk.download("punkt_tab")  # Download the punkt tokenizer models
    main(datadir=datadir)
