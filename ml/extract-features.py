#! /usr/bin/python3
"""
Tokenize sentence, returning tokens and span offsets
"""
import os
import sys
from os import listdir
from typing import List, Tuple, TypeAlias
from xml.dom.minidom import parse

import nltk
import nltk.tag
from nltk.tokenize import word_tokenize

Token: TypeAlias = Tuple[str, int, int]
EntitySpan: TypeAlias = Tuple[int, int, str]

nltk.download("averaged_perceptron_tagger_eng")
nltk.download("universal_tagset")


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
        features: List[str] = []

        features.append("form=" + word)  # Token form
        features.append("suf4=" + word[-3:])
        features.append("suf3=" + word[-3:])
        features.append("suf2=" + word[-2:])
        features.append("suf1=" + word[-1:])
        features.append("pre1=" + word[0:1])
        features.append("pre2=" + word[0:2])
        features.append("pre3=" + word[0:3])
        features.append("pre4=" + word[0:4])
        features.append("capitalized=" + str(word[0].isupper()))  # Is the first letter capitalized?
        features.append("uppercase=" + str(word.isupper()))  # Is the token all uppercase?
        features.append("hasdigit=" + str(any(c.isdigit() for c in word)))  # Does the token contain a digit?
        features.append("haspunct=" + str(any(c in ".,;:!?" for c in word)))  # Does the token contain punctuation?
        features.append("hashyphen=" + str("-" in word))  # Does the token contain a hyphen?
        features.append("length=" + str(len(word)))  # Length of the token
        features.append("form_lower=" + word.lower())  # Lowercase form of the token
        features.append("pos_tag=" + pos_tag)  # POS tag of the token

        if k == 0:
            features.append("BoS")
        if k == len(tokens) - 1:
            features.append("EoS")

        capitalization_pattern = ["0"] * 5  # Initialize capitalization pattern
        uppercase_pattern = ["0"] * 5  # Initialize uppercase pattern
        hasdigit_pattern = ["0"] * 5  # Initialize digit pattern
        hashyphen_pattern = ["0"] * 5  # Initialize hyphen pattern
        for j, i in enumerate(range(-2, 3)):
            if k + i < 0 or k + i >= len(tokens):
                continue
            window_word = tokens[k + i][0]
            if i != 0:
                features.append("window" + str(i) + "=" + window_word.lower())  # Context window features
                features.append("window" + str(i) + "-size=" + str(len(window_word)))  # Size of the context window
            if window_word[0].isupper():
                capitalization_pattern[j] = "1"
            if window_word.isupper():
                uppercase_pattern[j] = "1"
            if any(c.isdigit() for c in window_word):
                hasdigit_pattern[j] = "1"
            if "-" in window_word:
                hashyphen_pattern[j] = "1"

        features.append("neigh-capitalization=" + "".join(capitalization_pattern))  # Capitalization pattern
        features.append("neigh-uppercase=" + "".join(uppercase_pattern))  # Uppercase pattern
        features.append("neigh-hasdigit=" + "".join(hasdigit_pattern))  # Digit pattern
        features.append("neigh-hashyphen=" + "".join(hashyphen_pattern))  # Hyphen pattern

        result.append(features)

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
