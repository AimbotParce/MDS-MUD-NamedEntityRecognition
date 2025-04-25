import os
import xml.dom.minidom
from typing import Dict, List, Tuple, TypeAlias, TypedDict

import nltk
import nltk.tokenize

nltk.download("punkt_tab")


class TokenDict(TypedDict):
    lc_form: str  # lower case form
    form: str  # original form
    start: int  # start position in the sentence
    end: int  # end position in the sentence


class TaggedTokenDict(TokenDict):
    tag: str  # tag for the token (B-entity, I-entity, O)
    # B-entity: beginning of an entity
    # I-entity: inside an entity
    # O: outside an entity


EntityTagSpan: TypeAlias = Tuple[int, int, str]  # (start, end, tag) for an entity


class Dataset:
    #  Parse all XML files in given dir, and load a list of sentences.
    #  Each sentence is a list of tuples (word, start, end, tag)
    def __init__(self, datadir):
        self.data: Dict[str, TaggedTokenDict] = {}
        # process each file in directory
        for f in os.listdir(datadir):

            # parse XML file, obtaining a DOM tree
            tree = xml.dom.minidom.parse(datadir + "/" + f)

            # process each sentence in the file
            xml_sentences = tree.getElementsByTagName("sentence")
            for xml_sentence in xml_sentences:
                sentence_id = xml_sentence.attributes["id"].value  # get sentence id
                sentence_text = xml_sentence.attributes["text"].value  # get sentence text
                xml_entities = xml_sentence.getElementsByTagName("entity")

                spans: List[EntityTagSpan] = []
                for xml_entity in xml_entities:
                    # for discontinuous entities, we only get the first span
                    # (will not work, but there are few of them)
                    (start, end) = xml_entity.attributes["charOffset"].value.split(";")[0].split("-")
                    entity_type = xml_entity.attributes["type"].value
                    spans.append((int(start), int(end), entity_type))

                # convert the sentence to a list of tokens
                token_dicts = self.__tokenize(sentence_text)

                # add gold label to each token, and store it in self.data
                self.data[sentence_id] = []
                for token_dict in token_dicts:
                    # see if the token is part of an entity
                    token_dict["tag"] = self.__get_tag(token_dict, spans)
                    self.data[sentence_id].append(token_dict)

    # --------- tokenize sentence -----------
    # -- Tokenize sentence, returning tokens and span offsets
    def __tokenize(self, text: str) -> List[TokenDict]:
        offset = 0  # To optimize the search for the token in the text
        tokens: List[TokenDict] = []
        # word_tokenize splits words, taking into account punctuations, numbers, etc.
        for token in nltk.tokenize.word_tokenize(text):
            # Keep track of the position where each token should appear, and
            # store that information with the token
            offset = text.find(token, offset)
            tokens.append({"lc_form": token.lower(), "form": token, "start": offset, "end": offset + len(token) - 1})
            offset += len(token)

        return tokens

    # --------- get tag -----------
    #  Find out whether given token is marked as part of an entity in the XML
    def __get_tag(self, token: TokenDict, spans: List[EntityTagSpan]) -> str:
        for spanS, spanE, spanT in spans:
            if token["start"] == spanS and token["end"] <= spanE:
                return "B-" + spanT
            elif token["start"] >= spanS and token["end"] <= spanE:
                return "I-" + spanT
        return "O"

    # ---- iterator to get sentences in the data set
    def sentences(self):
        for sid in self.data:
            yield self.data[sid]

    # ---- iterator to get ids for sentence in the data set
    def sentence_ids(self):
        for sid in self.data:
            yield sid

    # ---- get one sentence by id
    def get_sentence(self, sid):
        return self.data[sid]

    # get sentences as token lists
    def tokens(self):
        """
        Get sentences as lists of tuples (sentence_id, word, start, end)
        """
        for sid in self.data:
            s: List[Tuple[str, str, int, int]] = []
            for w in self.data[sid]:
                s.append((sid, w["form"], w["start"], w["end"]))
            yield s
