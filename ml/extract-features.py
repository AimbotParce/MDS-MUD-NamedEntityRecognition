#! /usr/bin/python3
"""
Tokenize sentence, returning tokens and span offsets
"""
import os
import sys
from os import listdir
from pathlib import Path
from typing import Any, List, Tuple, TypeAlias
from xml.dom.minidom import parse

import nltk
import nltk.tag
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
        features: List[str] = []

        features.append("self-suffix-4=" + word[-3:])
        features.append("self-suffix-3=" + word[-3:])
        features.append("self-suffix-2=" + word[-2:])
        features.append("self-suffix-1=" + word[-1:])
        features.append("self-prefix-1=" + word[0:1])
        features.append("self-prefix-2=" + word[0:2])
        features.append("self-prefix-3=" + word[0:3])
        features.append("self-prefix-4=" + word[0:4])

        for i in range(-2, 3):
            # Add context window features
            neighbor_features = {
                "is-capitalized": "[NA]",  # Is the first letter capitalized?
                "is-uppercase": "[NA]",  # Is the token all uppercase?
                "has-digit": "[NA]",  # Does the token contain a digit?
                "has-punct": "[NA]",  # Does the token contain punctuation?
                "has-hyphen": "[NA]",  # Does the token contain a hyphen?
                "has-med-suffix": "[NA]",  # Does the token have a medical suffix?
                "med-suffix": "[NA]",  # Medical suffix of the token
                "has-med-form-word": "[NA]",  # Does the token have a medical form word?
                "med-form-word": "[NA]",  # Medical form word of the token
                "length": "[NA]",  # Length of the token
                "is-long": "[NA]",  # Is the token long?
                "form": "[NA]",  # Form of the token
                "form-lower": "[NA]",  # Lowercase form of the token
                "pos-tag": "[NA]",  # POS tag of the token
                "is-bos": "[NA]",  # Is the token at the beginning of the sentence?
                "is-eos": "[NA]",  # Is the token at the end of the sentence?
            }
            if not (k + i < 0 or k + i >= len(tokens)):
                (neighbor_word, neighbor_pos_tag) = pos_tags[k + i]
                neighbor_features["form"] = neighbor_word
                neighbor_features["form-lower"] = neighbor_word.lower()
                neighbor_features["length"] = str(len(neighbor_word))
                neighbor_features["is-long"] = str(len(neighbor_word) > 5)
                neighbor_features["pos-tag"] = neighbor_pos_tag
                neighbor_features["is-capitalized"] = str(neighbor_word[0].isupper())
                neighbor_features["is-uppercase"] = str(neighbor_word.isupper())
                neighbor_features["has-digit"] = str(any(c.isdigit() for c in neighbor_word))
                neighbor_features["has-punct"] = str(any(c in ".,;:!?" for c in neighbor_word))
                neighbor_features["has-hyphen"] = str("-" in neighbor_word)
                neighbor_features["is-bos"] = str(k + i == 0)
                neighbor_features["is-eos"] = str(k + i == len(tokens) - 1)
                for suffix in drug_suffixes:
                    if neighbor_word.lower().endswith(suffix):
                        neighbor_features["has-med-suffix"] = "True"
                        neighbor_features["med-suffix"] = suffix
                        break
                else:
                    neighbor_features["has-med-suffix"] = "False"
                    neighbor_features["med-suffix"] = "[NA]"
                for form_word in drug_form_words:
                    if neighbor_word.lower().endswith(form_word):
                        neighbor_features["has-med-form-word"] = "True"
                        neighbor_features["med-form-word"] = form_word
                        break
                else:
                    neighbor_features["has-med-form-word"] = "False"
                    neighbor_features["med-form-word"] = "[NA]"

            for key, val in neighbor_features.items():
                if i == 0:
                    features.append(f"self-{key}={val}")  # Add self features
                else:
                    if i < 0:
                        ctx_i = "l" + str(-i)
                    else:
                        ctx_i = "r" + str(i)
                    features.append(f"ctx-{ctx_i}-{key}={val}")  # Add context features

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
