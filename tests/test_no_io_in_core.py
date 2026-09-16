"""Ensure core algebra modules contain no I/O primitives."""

from pathlib import Path

CORE = ("cartan.py", "root_system.py", "element.py", "finite.py", "group.py", "weight_lattice.py", "symmetric_algebra.py", "schubert.py")
FORBIDDEN = ("print(", "input(", "argparse", "pathlib", "stdout", "stdin", "open(")


def test_core_modules_have_no_io():
    root = Path(__file__).resolve().parents[1] / "src" / "affineweyl"
    for name in CORE:
        text = (root / name).read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in text, f"{name} contains forbidden token {token!r}"
