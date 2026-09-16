"""Tests for the classical (finite) weight lattice P."""

from fractions import Fraction

import pytest

from affineweyl import AffineWeylGroup, FiniteRootSystem, WeightLattice
from affineweyl.cartan import _finite_cartan


@pytest.mark.parametrize(
    "series,n",
    [("A", 1), ("A", 2), ("A", 3), ("B", 2), ("B", 3), ("C", 2), ("C", 3), ("D", 4),
     ("E", 6), ("F", 4), ("G", 2)],
)
def test_fundamental_weights_pair_as_kronecker(series, n):
    rs = FiniteRootSystem.create(series, n)
    P = rs.weight_lattice
    for i in range(n):
        omega = P.fundamental_weight(i)
        assert omega == tuple(1 if j == i else 0 for j in range(n))
        for j in range(n):
            assert P.pairing_weight_coroot(omega, j) == (1 if i == j else 0)
            assert rs.pairing_weight_coroot(omega, j) == (1 if i == j else 0)


def test_group_weight_lattice_wiring():
    W = AffineWeylGroup.from_label("C~2")
    assert isinstance(W.weight_lattice, WeightLattice)
    assert W.weight_lattice is W.root_system.weight_lattice
    omega1 = W.fundamental_weight(1)
    assert W.pairing_weight_coroot(omega1, 1) == 1
    assert W.pairing_weight_coroot(omega1, 0) == 0


@pytest.mark.parametrize(
    "series,n,index",
    [
        ("A", 1, 2),
        ("A", 2, 3),
        ("A", 3, 4),
        ("A", 5, 6),
        ("B", 2, 2),
        ("B", 4, 2),
        ("C", 2, 2),
        ("C", 3, 2),
        ("D", 4, 4),
        ("D", 5, 4),
        ("E", 6, 3),
        ("E", 7, 2),
        ("E", 8, 1),
        ("F", 4, 1),
        ("G", 2, 1),
    ],
)
def test_index_P_mod_Q(series, n, index):
    P = FiniteRootSystem.create(series, n).weight_lattice
    assert P.index_P_mod_Q == index


@pytest.mark.parametrize("series,n", [("A", 2), ("A", 3), ("B", 2), ("C", 2), ("D", 4)])
def test_root_lattice_inside_weight_lattice(series, n):
    rs = FiniteRootSystem.create(series, n)
    P = rs.weight_lattice
    # Every simple root (and every positive root) is in Q ⊂ P
    for i in range(n):
        ei = tuple(1 if j == i else 0 for j in range(n))
        fund = P.from_simple_root_coords(ei)
        assert P.is_in_root_lattice(fund)
        # fund coords equal the i-th column of A? α_i = Σ_k a_{ki} ω_k
        # from_simple_root uses A @ ei = i-th column... wait A @ ei is i-th column of A
        # but α_i = Σ_k a_{ki} ω_k = i-th column. Yes.
        expected = tuple(rs.cartan[k][i] for k in range(n))
        assert fund == expected
    for alpha in rs.positive_roots:
        fund = P.from_simple_root_coords(alpha)
        assert P.is_in_root_lattice(fund)


def test_A_n_omega_not_always_in_Q():
    # For A_n, ω_0 ∉ Q when n >= 1 (index n+1 > 1)
    P = FiniteRootSystem.create("A", 2).weight_lattice
    assert not P.is_in_root_lattice((1, 0))
    assert not P.is_in_root_lattice((0, 1))
    # α_0 + α_1 = (1,1) in simple-root coords → fund = A(1,1) = (1,1)
    assert P.is_in_root_lattice(P.from_simple_root_coords((1, 1)))


@pytest.mark.parametrize("series,n", [("A", 1), ("A", 3), ("B", 3), ("C", 2), ("D", 4), ("G", 2)])
def test_roundtrip_fund_and_simple_root_coords(series, n):
    P = FiniteRootSystem.create(series, n).weight_lattice
    samples = [
        tuple(1 if j == i else 0 for j in range(n))  # simple roots
        for i in range(n)
    ]
    samples.append(tuple(1 for _ in range(n)))
    samples.append(tuple(range(1, n + 1)))
    for mu in samples:
        fund = P.from_simple_root_coords(mu)
        back = P.to_simple_root_coords(fund)
        assert all(c.denominator == 1 for c in back)
        assert tuple(int(c) for c in back) == mu


def test_to_simple_root_coords_rational_for_fundamental_weight():
    # A_2: A^{-1} = (1/3) [[2,1],[1,2]], so ω_0 ↦ (2/3, 1/3)
    P = FiniteRootSystem.create("A", 2).weight_lattice
    x = P.to_simple_root_coords((1, 0))
    assert x == (Fraction(2, 3), Fraction(1, 3))
    y = P.to_simple_root_coords((0, 1))
    assert y == (Fraction(1, 3), Fraction(2, 3))


def test_B2_C2_conversions():
    # B2 Cartan [[2,-1],[-2,2]]; C2 [[2,-2],[-1,2]]
    PB = FiniteRootSystem.create("B", 2).weight_lattice
    PC = FiniteRootSystem.create("C", 2).weight_lattice
    assert PB.from_simple_root_coords((1, 0)) == (2, -2)
    assert PB.from_simple_root_coords((0, 1)) == (-1, 2)
    assert PC.from_simple_root_coords((1, 0)) == (2, -1)
    assert PC.from_simple_root_coords((0, 1)) == (-2, 2)
    assert PB.index_P_mod_Q == 2
    assert PC.index_P_mod_Q == 2


def test_coweight_helpers():
    P = FiniteRootSystem.create("A", 2).weight_lattice
    assert P.fundamental_coweight(0) == (1, 0)
    assert P.pairing_root_coweight((1, 0), 0) == 1
    assert P.is_in_coroot_lattice(P.from_simple_coroot_coords((1, 0)))
    assert not P.is_in_coroot_lattice((1, 0))


def test_invalid_weight_length_and_index():
    P = FiniteRootSystem.create("A", 2).weight_lattice
    with pytest.raises(ValueError):
        P.pairing_weight_coroot((1,), 0)
    with pytest.raises(IndexError):
        P.fundamental_weight(2)
    with pytest.raises(IndexError):
        P.pairing_weight_coroot((1, 0), 5)
    with pytest.raises(TypeError):
        P.from_simple_root_coords((1.0, 0))  # type: ignore[arg-type]


def test_cartan_matches_package():
    rs = FiniteRootSystem.create("D", 4)
    assert rs.cartan == _finite_cartan("D", 4)
    assert rs.weight_lattice.cartan == rs.cartan
