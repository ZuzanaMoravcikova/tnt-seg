# -------------------------------------------------------------
# Author: Zuzana Moravčíková
# Project: Deep-Learning-Based Segmentation of Tunneling Nanotubes in Volumetric Bioimage Data
# Year: 2025
# Contact: 514286@mail.muni.cz
# License: MIT
# -------------------------------------------------------------

import argparse
from typing import Tuple


def int_triplet(s: str) -> Tuple[int, int, int]:
    """Parse a string representing ``Z,Y,X`` integer triplets.

    Accepts input formats such as:
    - ``"7,128,128"``
    - ``"(7,128,128)"``
    - ``"[7,128,128]"``

    Parameters
    ----------
    s: Input string to be parsed.

    Returns
    -------
    triplet: Three integers representing ``(Z, Y, X)``.
    """
    s = s.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = [p for p in s.split(",") if p != ""]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated integers, e.g. 7,128,128")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def int_pair(s: str) -> Tuple[int, int]:
    """Parse a string representing ``H,W`` integer pairs.

    Accepts input formats such as:
    - ``"64,128"``
    - ``"(64,128)"``
    - ``"[64,128]"``

    Parameters
    ----------
    s: Input string to be parsed.

    Returns
    -------
    pair: Two integers representing ``(H, W)``.
    """
    s = s.strip().replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = [p for p in s.split(",") if p != ""]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 2 comma-separated integers, e.g. 64,128")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]
