#! /usr/bin/python3
"""
Tokenize sentence, returning tokens and span offsets
"""
import os
import re
import sys
from collections import defaultdict
from os import listdir
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, TypeAlias
from xml.dom.minidom import parse

import nltk  # type: ignore
import nltk.tag  # type: ignore
import numpy as np
from nltk.corpus import stopwords  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

from chemdataextractor.nlp.tokenize import ChemWordTokenizer  # type: ignore

Token: TypeAlias = Tuple[str, int, int]
EntitySpan: TypeAlias = Tuple[int, int, str]

nltk.download("averaged_perceptron_tagger_eng")
nltk.download("universal_tagset")
nltk.download("stopwords")
nltk.download("punkt_tab")  # Download the punkt tokenizer models

cwt = ChemWordTokenizer()


def read_list(filename: str) -> Generator[str, None, None]:
    """
    Read a list of strings from a file, stripping whitespace and newlines.
    Args:
        filename (str): The name of the file to read.
    Returns:
        List[str]: A list of strings read from the file.
    """
    with open(filename, "r") as f:
        return (line.strip() for line in f.readlines())


def data_path(filename: str) -> Path:
    """
    Get the absolute path of a file in the data directory.
    Args:
        filename (str): The name of the file.
    Returns:
        file_path (Path): The absolute path of the file.
    """
    return (Path(__file__).parent.parent / "data" / filename).resolve()


def load_gazetteer(filename: Path) -> set[str]:
    """Reads a file into a set of lowercase strings (one per line)."""
    print(f"Loading gazetteer: {filename.name}...", file=sys.stderr)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            # Lowercase and strip whitespace for consistent matching
            items = {line.strip().lower() for line in f if line.strip()}
        print(f"Loaded {len(items)} unique entries.", file=sys.stderr)
        return items
    except FileNotFoundError:
        print(f"Warning: Gazetteer file not found: {filename}", file=sys.stderr)
        return set()  # Return empty set if file doesn't exist
    except Exception as e:
        print(f"Error loading gazetteer {filename}: {e}", file=sys.stderr)
        return set()


stopwords = set(stopwords.words("english"))  # Get English stop words from nltk

drug_suffixes = list(
    read_list(data_path("med-suffixes.txt"))
)  # Read drug suffixes from a file
drug_prefixes = list(
    read_list(data_path("med-prefixes.txt"))
)  # Read drug suffixes from a file
drug_form_words = list(
    read_list(data_path("med-form-words.txt"))
)  # Read drug form words from a file
drug_n_gaz = load_gazetteer(data_path("correct_drug_n.txt"))
fp_neg = load_gazetteer(data_path("drug_n_neg.txt"))

# Read hazardous substances from a file and convert them into a set of strings for faster lookup
hazardous_substances = set[str]()
for sentence in read_list(data_path("hazardous-substances.txt")):
    # Remove stop words
    for word in nltk.tokenize.word_tokenize(sentence):
        # Note: Keep in mind that each line may contain words that are not actually
        # hazardous substances, so let's remove stop words with nltk
        if word.lower() not in stopwords:
            hazardous_substances.add(word.lower())


word_drug_pattern = re.compile(
    r"\b(?=\w*[A-Za-z])\w+\b"
)  # All alphanumeric words (except only digits)
drug_categories: defaultdict[str, set[str]] = defaultdict(set)

for line in read_list(data_path("drugs.txt")):
    # This file is a bit different. It contains technical terms for drugs, as well
    # as their categorization, separated by "|"
    drug, tag = line.split("|")
    # Extract all alphanumeric words from the drug name
    for word in word_drug_pattern.findall(drug):
        # Add the word to the set of drug words for the corresponding tag
        if len(word) < 3:  # Ignore words with less than 3 characters
            continue
        drug_categories[tag].add(word.lower())

# Load gazeteers
# pubchem_set = load_gazetteer(data_path("pubchem_synonyms.txt"))
# chebi_set = load_gazetteer(data_path("chebi_synonyms.txt"))
# drugbank_set = load_gazetteer(data_path("drugbank_synonyms.txt"))


# Advanced REGEX stuff for enhanced drug_n performance
# Locant Patterns
locant_pattern_start = re.compile(r"^\d+([,]\d+)*-")
locant_pattern_internal = re.compile(r".*-\d+([,]\d+)*-.*")  # Search anywhere
locant_pattern_end = re.compile(r"-\d+[a-zA-Z]?$")  # Search for suffix

# Stereo/Isomer Indicators
stereo_paren_start = re.compile(
    r"^\([\+\-\w]+\)"
)  # Matches (+), (-), (R), (S), (rac)- etc.
stereo_letter_prefix_start = re.compile(r"^[DdEeLlRrSs]-")  # Matches D-, L-, R-, S-
stereo_greek_prefix_start = re.compile(
    r"^(alpha|beta|gamma|delta|omega)-", re.IGNORECASE
)

# Attachment Points
attach_point_start = re.compile(r"^[NOPS]-", re.IGNORECASE)  # N-, O-, P-, S-

# Common Chemical Suffixes/Fragments (using search to find at end)
# This list is from the report, might need curation
chem_suffix_pattern = re.compile(
    r"(amine|imine|azole|idine|azine|oxane|anone|anoic|acetyl|phenyl|methyl|ethyl|propyl|butyl|fluoro|chloro|bromo|iodo|hydroxy|methoxy|ethoxy|cyano|nitro|amino|silyl|uracil|thymine|cytosine|guanine|adenine|purine|pyridine|pyran|furan|thiophene|amide|imide|acid|ate|ol|al|one|ene|yne)$",
    re.IGNORECASE,
)

# Specific Code Formats
code_format_az_num = re.compile(r"^[A-Z]+[-]?\d+$")
code_format_num_az = re.compile(r"^\d+-?[A-Z]+$")

# Bracketed/Complex Structures
bracketed_paren_whole = re.compile(r"^\([^)]+\)$")  # Token is exactly (...)
bracketed_square_contains = re.compile(r"\[.*\]")  # Token contains [...]

# Potentially good regex for drug_n
rx_lead_num_sep = re.compile(r"^\d+([,.]\d+)*[-(\[]")
rx_internal_num_sep = re.compile(
    r"\b[A-Za-z]+\d*[- ,.]\d+\b|\b\d+[- ,.]\d*[A-Za-z]+\b|\b\d+,\d+-\w"
)
rx_brackets_parens = re.compile(r"[\(\[]\w*[-]?\w*[\)\]]|\w\(\w+\)")
rx_potential_code = re.compile(
    r"\b(?:[A-Z]{2,}\d?|[A-Z]*\d+[A-Z]+|[A-Z]+-\d+|[A-Z]+\d+-|[A-Z\d]+-[A-Z\d]+|\d+-[A-Z]+)(?:-\w+)?\b"
)
rx_chem_prefix = re.compile(
    r"\b(des|nor|dehydro|[bh]ydroxy|methyl|ethyl|phenyl|fluoro|chloro|bromo|iodo|acetyl|carbo|amino|oxo|alpha|beta|gamma|delta|omega|cis|trans|para|meta|ortho|[NO]-)\w+",
    re.IGNORECASE,
)
rx_chem_suffix = re.compile(
    r"\w+(?:yl|ol|one|ate|ide|ase|ine|azole|idine|azine|oic|al|ene|yne|oxin|idine|amide|tannin|saponin|oside)\b",
    re.IGNORECASE,
)
rx_digit_hyphen = re.compile(r"^(?=.*\d)(?=.*-).+$")
rx_two_non_alnum = re.compile(r"(?:[^A-Za-z0-9\s].*){2}")

drug_n_regexes_list = [
    ("lead_num_sep", rx_lead_num_sep),
    ("internal_num_sep", rx_internal_num_sep),
    ("brackets_parens", rx_brackets_parens),
    ("potential_code", rx_potential_code),
    ("chem_prefix", rx_chem_prefix),
    ("chem_suffix", rx_chem_suffix),
    ("digit_hyphen", rx_digit_hyphen),
    ("two_non_alnum", rx_two_non_alnum),
]


def tokenize(txt: str) -> List[Token]:
    """
    Tokenize a text string using ChemWordTokenizer, returning a list of tokens
    and their start and end positions (inclusive end).

    Args:
        txt (str): The text string to tokenize.

    Returns:
        List[Token]: A list of tuples, each containing a token string, its
                     start position, and its inclusive end position.
    """

    tks: List[Token] = []

    offset = 0
    for t in cwt.tokenize(txt):
        offset = txt.find(t, offset)
        tks.append((t, offset, offset + len(t) - 1))
        offset += len(t)

    # for span in spans:
    #     token_text = txt[span.start : span.end]  # Extract text using span offsets
    #     # Store token, start offset, and INCLUSIVE end offset (span.end - 1)
    #     # This matches the format expected by the original get_tag function
    #     # based on the offset calculation: offset + len(t) - 1
    #     tks.append((token_text, span.start, span.end - 1))

    return tks


# def tokenize(txt: str):
#     """
#     Tokenize a text string, returning a list of tokens and their
#     start and end positions in the original string.
#
#     Args:
#         txt (str): The text string to tokenize.
#     Returns:
#         List[Token]: A list of tuples, each containing a token and its
#                      start and end positions in the original string.
#     """
#
#     offset = 0
#     tks: List[Token] = []
#     for t in word_tokenize(
#         txt
#     ):  # word_tokenize splits words, taking into account punctuations, numbers, etc.
#         # Keep track of the position where each token should appear, and
#         # store that information with the token
#         offset = txt.find(t, offset)
#         tks.append((t, offset, offset + len(t) - 1))
#         offset += len(t)
#     return tks  # tks is a list of triples (word,start,end)
#


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
    pos_tags = nltk.tag.pos_tag(
        [t[0] for t in tokens], tagset="universal"
    )  # Get POS tags for the tokens
    assert len(tokens) == len(pos_tags), "Mismatch between tokens and POS tags length"
    for k, (word, pos_tag) in enumerate(pos_tags):
        features: Dict[str, str] = {}

        features["form"] = word  # Token form
        features["form-lower"] = word.lower()  # Lowercase form of the token
        features["is-capitalized"] = str(
            word[0].isupper()
        )  # Is the first letter capitalized?
        features["is-uppercase"] = str(word.isupper())  # Is the token all uppercase?
        features["has-digit"] = str(
            any(c.isdigit() for c in word)
        )  # Does the token contain a digit?
        features["has-punct"] = str(
            any(c in ".,;:!?" for c in word)
        )  # Does the token contain punctuation?
        features["has-hyphen"] = str("-" in word)  # Does the token contain a hyphen?
        features["length"] = str(len(word))  # Length of the token
        features["is-long"] = str(len(word) > 5)  # Is the token long?
        features["pos-tag"] = pos_tag  # POS tag of the token
        features["is-bos"] = str(
            k == 0
        )  # Is the token at the beginning of the sentence?
        features["is-eos"] = str(
            k == len(tokens) - 1
        )  # Is the token at the end of the sentence?
        features["is-stopword"] = str(
            word.lower() in stopwords
        )  # Is the token a stop word?
        features["is-hazardous"] = str(
            word.lower() in hazardous_substances
        )  # Is the token a hazardous substance?

        features["is-drug"] = "False"
        features["drug-type"] = "[NA]"

        # # Gazeteer features
        # features["feat_in_pubchem"] = str(word.lower() in pubchem_set)
        # features["feat_in_chebi"] = str(word.lower() in chebi_set)
        # features["feat_in_drugbank"] = str(word.lower() in drugbank_set)

        # Advanced REGEX features
        features["feat_regex_locant_start"] = str(
            bool(locant_pattern_start.match(word))
        )
        features["feat_regex_locant_internal"] = str(
            bool(locant_pattern_internal.search(word))
        )
        features["feat_regex_locant_end"] = str(bool(locant_pattern_end.search(word)))
        features["feat_regex_stereo_paren"] = str(bool(stereo_paren_start.match(word)))
        features["feat_regex_stereo_letter"] = str(
            bool(stereo_letter_prefix_start.match(word))
        )
        features["feat_regex_stereo_greek"] = str(
            bool(stereo_greek_prefix_start.match(word))
        )
        features["feat_regex_attach_point"] = str(bool(attach_point_start.match(word)))
        features["feat_regex_chem_suffix"] = str(bool(chem_suffix_pattern.search(word)))
        features["feat_regex_code_az_num"] = str(bool(code_format_az_num.match(word)))
        features["feat_regex_code_num_az"] = str(bool(code_format_num_az.match(word)))
        features["feat_regex_bracket_paren"] = str(
            bool(bracketed_paren_whole.match(word))
        )
        features["feat_regex_bracket_square"] = str(
            bool(bracketed_square_contains.search(word))
        )

        # drug_n regexes
        any_drug_n_regex_matched = False
        # Iterate through the list of compiled regexes and their names
        for name_suffix, compiled_regex in drug_n_regexes_list:
            # Use search() to find the pattern anywhere in the word
            match_result = compiled_regex.search(word)
            is_match = bool(match_result)
            # Add a feature like "match_rx_lead_num_sep=True"
            # features[f"match_rx_{name_suffix}"] = str(is_match)
            if is_match:
                any_drug_n_regex_matched = True  # Set the flag if any regex matches

        rx_hits = sum(1 for _, rx in drug_n_regexes_list if rx.search(word))
        features["drug_n_regex_hits"] = str(rx_hits)  # categorical feature
        features["drug_n_regex_ge2"] = str(rx_hits >= 2)  # binary shortcut

        # Add the combined feature: True if any of the drug_n regexes matched
        features["potential_drug_n"] = str(any_drug_n_regex_matched)

        # Character n-gram features
        # Consider lengths 3 and 4, (maybe 5??)
        char_ngram_lengths = [3, 4]
        # Add boundary markers for prefix/suffix detection
        word_lower_padded = f"^{word.lower()}$"
        for n in char_ngram_lengths:
            if len(word_lower_padded) >= n:
                for i in range(len(word_lower_padded) - n + 1):
                    ngram = word_lower_padded[i : i + n]
                    features[f"char_ngram_{n}_{ngram}"] = "True"

        for tag, drug_words in drug_categories.items():
            if word in drug_words:
                features["is-drug"] = "True"
                features["drug-type"] = tag
                break

        # # Fuzzy match for drug names
        # for tag, drug_words in drug_categories.items():
        #     for drug_word in drug_words:
        #         if drug_word in word.lower() or word.lower() in drug_word:
        #             similarity = len(word.lower()) / max(
        #                 len(drug_word), len(word.lower())
        #             )
        #             if similarity > 0.75:
        #                 features["fuzzy-drug-match"] = "True"
        #                 break

        for suffix in drug_suffixes:
            if word.lower().endswith(suffix):
                features["has-med-suffix"] = "True"
                features["med-suffix"] = suffix
                break
        else:
            features["has-med-suffix"] = "False"
            features["med-suffix"] = "[NA]"

        for prefix in drug_prefixes:
            if word.lower().startswith(prefix):
                features["has-med-prefix"] = "True"
                features["med-prefix"] = prefix
                break
        else:
            features["has-med-prefix"] = "False"
            features["med-prefix"] = "[NA]"

        for form_word in drug_form_words:
            if word.lower() == form_word:
                features["is-med-form-word"] = "True"
                break
        else:
            features["is-med-form-word"] = "False"

        # A) positive gazetteer
        features["in-drug_n-gaz"] = str(word.lower() in drug_n_gaz)
        # B) frequent false-positives
        features["is-drug_n-fp"] = str(word.lower() in fp_neg)

        # Context window features
        context_size = 5
        context_info = defaultdict(lambda: np.zeros(context_size, dtype=bool))
        for j, i in enumerate(
            range(-(context_size // 2), context_size - (context_size // 2))
        ):
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
        sentence_elements = tree.getElementsByTagName(
            "sentence"
        )  # Get all sentences in the file
        for sentence in sentence_elements:
            sentence_id = sentence.attributes["id"].value  # Get sentence id
            spans: List[EntitySpan] = []
            sentence_content = sentence.attributes["text"].value  # Get sentence text
            entity_elements = sentence.getElementsByTagName(
                "entity"
            )  # Get pre-annotated entities
            for entity in entity_elements:
                # For discontinuous entities, we only get the first span
                # (will not work, but there are few of them)
                (start, end) = (
                    entity.attributes["charOffset"].value.split(";")[0].split("-")
                )
                entity_type = entity.attributes["type"].value
                spans.append((int(start), int(end), entity_type))

            tokens = tokenize(
                sentence_content
            )  # Convert the sentence to a list of tokens
            features = extract_features(tokens)  # Extract sentence features

            # Print features in format expected by crfsuite trainer
            for i in range(0, len(tokens)):
                # See if the token is part of an entity
                tag = get_tag(tokens[i], spans)
                print(
                    sentence_id,
                    tokens[i][0],
                    tokens[i][1],
                    tokens[i][2],
                    tag,
                    "\t".join(features[i]),
                    sep="\t",
                )

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
