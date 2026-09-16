"""Thin CLI for affineweyl — all logic delegated to the library."""

from __future__ import annotations

import argparse
import sys

from .cartan import list_supported_examples
from .group import AffineWeylGroup
from .io import format_element, parse_element, parse_word


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="affineweyl",
        description="Affine Weyl group calculator (untwisted types).",
    )
    p.add_argument(
        "-t",
        "--type",
        default="A~2",
        help="Affine type label, e.g. A~2, tilde_C3, E~6 (default: A~2)",
    )
    sub = p.add_subparsers(dest="cmd")

    mul = sub.add_parser("mul", help="Multiply a word and print a reduced expression")
    mul.add_argument("word", nargs="+", help="Word tokens, e.g. s0 s1 s0")

    length = sub.add_parser("length", help="Length of a word's product")
    length.add_argument("word", nargs="+", help="Word tokens")

    red = sub.add_parser("reduced", help="Check whether a word is reduced")
    red.add_argument("word", nargs="+", help="Word tokens")

    enum = sub.add_parser("enumerate", help="List elements up to a given length")
    enum.add_argument("max_length", type=int)

    types = sub.add_parser("types", help="List example supported type labels")

    info = sub.add_parser("info", help="Show Cartan/Coxeter data for a type")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 0

    if args.cmd == "types":
        for lab in list_supported_examples():
            print(lab)
        return 0

    group = AffineWeylGroup.from_label(args.type)

    if args.cmd == "info":
        print(f"type: {group.label}")
        print(f"affine rank: {group.affine_rank}")
        print(f"finite rank: {group.finite_rank}")
        print("Cartan matrix:")
        for row in group.cartan_matrix:
            print(" ", row)
        print("Coxeter matrix:")
        for row in group.coxeter_matrix:
            print(" ", row)
        return 0

    if args.cmd == "enumerate":
        elts = group.elements_up_to_length(args.max_length)
        for e in elts:
            print(format_element(e))
        print(f"count: {len(elts)}")
        return 0

    word_str = " ".join(args.word)
    word = parse_word(word_str)
    # validate indices
    for i in word:
        if i < 0 or i >= group.affine_rank:
            print(
                f"error: index {i} out of range for {group.label} "
                f"(need 0..{group.affine_rank - 1})",
                file=sys.stderr,
            )
            return 2
    elt = group.from_word(word)

    if args.cmd == "mul":
        print(format_element(elt))
        return 0
    if args.cmd == "length":
        print(elt.length)
        return 0
    if args.cmd == "reduced":
        print("reduced" if group.is_reduced(word) else "not reduced")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
