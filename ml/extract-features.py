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
from nltk.tokenize import word_tokenize

Token: TypeAlias = Tuple[str, int, int]
EntitySpan: TypeAlias = Tuple[int, int, str]


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
    for k in range(0, len(tokens)):
        tokenFeatures: List[str] = []
        t = tokens[k][0]

        tokenFeatures.append("form=" + t)  # Token form
        tokenFeatures.append("suf3=" + t[-3:])

        if k > 0:
            tPrev = tokens[k - 1][0]
            tokenFeatures.append("formPrev=" + tPrev)  # Token form of previous token
            tokenFeatures.append("suf3Prev=" + tPrev[-3:])  # Suffix of previous token ???
        else:
            tokenFeatures.append("BoS")  # Beginning of Sentence

        if k < len(tokens) - 1:
            tNext = tokens[k + 1][0]
            tokenFeatures.append("formNext=" + tNext)  # Token form of next token
            tokenFeatures.append("suf3Next=" + tNext[-3:])  # Suffix of previous token ???
        else:
            tokenFeatures.append("EoS")  # End of Sentence

        result.append(tokenFeatures)

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
