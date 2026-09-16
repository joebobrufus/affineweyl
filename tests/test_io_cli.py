import subprocess
import sys

from affineweyl import AffineWeylGroup, format_element, parse_element, parse_word
from affineweyl.io import format_word


def test_parse_word_variants():
    assert parse_word("s0 s1 s0") == (0, 1, 0)
    assert parse_word("s_0*s_1") == (0, 1)
    assert parse_word("s0*s1*s2") == (0, 1, 2)
    assert parse_word("0 1 0") == (0, 1, 0)
    assert parse_word("1") == ()
    assert parse_word("id") == ()


def test_parse_element_roundtrip():
    W = AffineWeylGroup.from_label("A~2")
    e = parse_element(W, "s0*s1*s0")
    assert e == W.from_word([0, 1, 0])
    text = format_element(e)
    assert "length 3" in text
    assert format_word(e.reduced_word(), style="star").startswith("s")


def test_cli_mul():
    proc = subprocess.run(
        [sys.executable, "-m", "affineweyl", "-t", "A~2", "mul", "s0", "s1", "s0"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "length 3" in proc.stdout


def test_cli_types():
    proc = subprocess.run(
        [sys.executable, "-m", "affineweyl", "types"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "A~1" in proc.stdout


def test_cli_reduced():
    proc = subprocess.run(
        [sys.executable, "-m", "affineweyl", "-t", "A~1", "reduced", "s0", "s0"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "not reduced" in proc.stdout


def test_cli_info():
    proc = subprocess.run(
        [sys.executable, "-m", "affineweyl", "-t", "G~2", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "G~2" in proc.stdout
