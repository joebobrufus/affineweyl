"""Parsing and formatting for affine Weyl elements (I/O layer)."""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .element import AffineWeylElement
from .group import AffineWeylGroup

_TOKEN = re.compile(
    r"""
    s_?(?P<idx>\d+)      # s0, s_0, s12
    | [?*^·•]?           # optional separators consumed loosely
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_word(text: str) -> Tuple[int, ...]:
    """Parse a word string into simple indices.

    Accepts forms such as::

        s0 s1 s0
        s_0*s_1*s_0
        s0*s1
        s0s1s0
        0 1 0
    """
    text = text.strip()
    if not text or text in {"1", "e", "id", "I"}:
        return ()
    # Prefer explicit s-tokens
    indices = [int(m.group(1)) for m in re.finditer(r"s_?(\d+)", text, re.I)]
    if indices:
        return tuple(indices)
    # Fallback: bare integers
    parts = re.split(r"[\s,*;·•x×]+", text)
    out: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"Cannot parse token {p!r} in word {text!r}")
        out.append(int(p))
    return tuple(out)


def parse_element(group: AffineWeylGroup, text: str) -> AffineWeylElement:
    """Parse a word string into an element of ``group``."""
    return group.from_word(parse_word(text))


def format_word(word: Sequence[int], style: str = "space") -> str:
    """Format a word. Styles: ``space`` -> ``s0 s1``, ``star`` -> ``s0*s1``, ``underscore``."""
    if not word:
        return "1"
    if style == "star":
        return "*".join(f"s{i}" for i in word)
    if style == "underscore":
        return "*".join(f"s_{i}" for i in word)
    return " ".join(f"s{i}" for i in word)


def format_element(element: AffineWeylElement, style: str = "space") -> str:
    """Format an element as a reduced word plus length annotation."""
    word = element.reduced_word()
    body = format_word(word, style=style)
    return f"{body}  (length {element.length})"


__all__ = [
    "parse_word",
    "parse_element",
    "format_word",
    "format_element",
]
