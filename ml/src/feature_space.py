#!/usr/bin/env python3
from typing import Generator, List, TextIO, Tuple, overload, Union


class SentenceYielder:
    """
    Yield data columns from the tokens of the sentences from a feature file.
    The feature file is a tab-separated file with the following format:
    ```
    sid  form  span_start  span_end  tag  feature1  feature2 ... featureN
    ```
    Where each line represents a token in a sentence, and the sentences
    are separated by empty lines.

    This class provides a single method __getitem__ that allows you to specify
    two slices or indices to extract from each tab-separated line of the file.

    The result is a generator that, for each sentence, yields two lists:
    - The first list contains the feature/s from the first specified column/s.
    - The second list contains the feature/s from the second specified column/s.

    Example usage:
    ```python
    with open("path/to/feature_file.txt", "r") as fd:
        instance_yielder = SentenceYielder(fd)
        for xseq, yseq in instance_yielder[0:2, 4]:
            # Each iteration corresponds to a sentence.
            print(xseq)  # Prints the features from columns 0 to 2
            print(yseq)  # Prints the labels from column 4
    ```
    """

    def __init__(self, fd: TextIO):
        self.fd = fd

    @overload
    def __getitem__(self, indices: Tuple[int, int]) -> Generator[Tuple[List[str], List[str]], None, None]: ...
    @overload
    def __getitem__(
        self, indices: Tuple[slice, slice]
    ) -> Generator[Tuple[List[List[str]], List[List[str]]], None, None]: ...
    @overload
    def __getitem__(self, indices: Tuple[int, slice]) -> Generator[Tuple[List[str], List[List[str]]], None, None]: ...
    @overload
    def __getitem__(self, indices: Tuple[slice, int]) -> Generator[Tuple[List[List[str]], List[str]], None, None]: ...

    def __getitem__(
        self, indices: Tuple[Union[int, slice], Union[int, slice]]
    ) -> Generator[Tuple[Union[List[List[str]], List[str]], Union[List[List[str]], List[str]]], None, None]:
        xslices, yslices = indices
        xseq = []
        yseq = []

        for line in self.fd:
            line = line.strip("\n")
            if not line:
                # An empty line means the end of a sentence.
                # Return accumulated sequences, and reinitialize.
                if xseq and yseq:
                    yield xseq, yseq

                xseq = []
                yseq = []
                continue

            # Split the line with TAB characters.
            fields = line.split("\t")

            # Append the item features to the item sequence.
            # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
            xseq.append(fields[xslices])
            yseq.append(fields[yslices])
