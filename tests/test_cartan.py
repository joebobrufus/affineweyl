import pytest

from affineweyl.cartan import (
    canonical_label,
    cartan_A,
    cartan_B,
    cartan_C,
    coxeter_from_cartan,
    parse_affine_type,
    validate_rank,
)
from affineweyl.group import AffineWeylGroup


def test_parse_labels():
    assert parse_affine_type("A~3") == ("A", 3)
    assert parse_affine_type("tilde_A3") == ("A", 3)
    assert parse_affine_type("tildeA_2") == ("A", 2)
    assert parse_affine_type("C~2") == ("C", 2)
    assert parse_affine_type("E~6") == ("E", 6)
    assert parse_affine_type("a~1") == ("A", 1)


def test_canonical():
    assert canonical_label("a", 2) == "A~2"


def test_cartan_A2_finite():
    A = cartan_A(2)
    assert A == ((2, -1), (-1, 2))


def test_coxeter_from_cartan_A():
    m = coxeter_from_cartan(cartan_A(2))
    assert m[0][1] == 3


def test_coxeter_B_double_bond():
    m = coxeter_from_cartan(cartan_B(2))
    assert m[0][1] == 4


def test_coxeter_C_double_bond():
    m = coxeter_from_cartan(cartan_C(2))
    assert m[0][1] == 4


def test_validate():
    with pytest.raises(ValueError):
        validate_rank("D", 3)
    with pytest.raises(ValueError):
        AffineWeylGroup.from_label("B~1")


def test_affine_cartan_A1():
    W = AffineWeylGroup.from_label("A~1")
    C = W.cartan_matrix
    assert len(C) == 2
    assert C[0][0] == 2 and C[1][1] == 2
    # A~1: both off-diagonals -2
    assert C[0][1] == -2 and C[1][0] == -2
