#!/usr/bin/env python3
from typing import Generator, List, TextIO, Tuple, overload


class InstanceYielder:
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
        self, indices: Tuple[int | slice, int | slice]
    ) -> Generator[Tuple[List[List[str]] | List[str], List[List[str]] | List[str]], None, None]:
        xslices, yslices = indices
        xseq = []
        yseq = []

        for line in self.fd:
            line = line.strip("\n")
            if not line:
                # An empty line means the end of a sentence.
                # Return accumulated sequences, and reinitialize.
                yield xseq, yseq
                xseq = []
                yseq = []
                continue

            # Split the line with TAB characters.
            fields = line.split("\t")

            # Append the item features to the item sequence.
            # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
            xseq.append(fields[xslices])
            yseq.append(fields[yslices])  # Append the label to the label sequence.
