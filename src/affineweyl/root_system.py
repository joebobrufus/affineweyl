"""Finite root systems from Cartan matrices.

Roots and coroots are integer coordinate tuples in the simple (co)root bases.
No I/O in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import Iterable, List, Sequence, Tuple

from .cartan import Matrix, _finite_cartan, coxeter_from_cartan, transpose

Vector = Tuple[int, ...]


def _vec_add(a: Vector, b: Vector) -> Vector:
    return tuple(x + y for x, y in zip(a, b))


def _vec_sub(a: Vector, b: Vector) -> Vector:
    return tuple(x - y for x, y in zip(a, b))


def _vec_scale(a: Vector, k: int) -> Vector:
    return tuple(k * x for x in a)


def _dot(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(x * y for x, y in zip(a, b))


def _matmul(M: Matrix, v: Vector) -> Vector:
    return tuple(sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M)))


def _matmatmul(A: Matrix, B: Matrix) -> Matrix:
    n = len(A)
    m = len(B[0])
    p = len(B)
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(p)) for j in range(m))
        for i in range(n)
    )


def _identity(n: int) -> Matrix:
    return tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))


def reflection_on_roots(cartan: Matrix, i: int) -> Matrix:
    """Matrix of s_i on simple-root coordinates: s_i(α_j) = α_j - a_{ij} α_i."""
    n = len(cartan)
    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            val = 1 if r == c else 0
            if r == i:
                val -= cartan[i][c]
            row.append(val)
        rows.append(tuple(row))
    return tuple(rows)


def reflection_on_coroots(cartan: Matrix, i: int) -> Matrix:
    """Matrix of s_i on simple-coroot coordinates: s_i(α_j∨) = α_j∨ - a_{ji} α_i∨."""
    n = len(cartan)
    rows = []
    for r in range(n):
        row = []
        for c in range(n):
            val = 1 if r == c else 0
            if r == i:
                val -= cartan[c][i]
            row.append(val)
        rows.append(tuple(row))
    return tuple(rows)


def _height(v: Vector) -> int:
    return sum(v)


def _enumerate_positive_roots(cartan: Matrix) -> Tuple[Vector, ...]:
    """All positive roots as simple-root coordinate tuples, sorted by height then lex."""
    n = len(cartan)
    simples = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)]
    mats = [reflection_on_roots(cartan, i) for i in range(n)]
    seen = set(simples)
    queue = list(simples)
    while queue:
        beta = queue.pop()
        for M in mats:
            gamma = _matmul(M, beta)
            if _height(gamma) > 0 and gamma not in seen:
                seen.add(gamma)
                queue.append(gamma)
    return tuple(sorted(seen, key=lambda v: (_height(v), v)))


def _simple_root_lengths_squared(cartan: Matrix) -> Tuple[Fraction, ...]:
    """Squared lengths of simple roots, normalized so some long root has length^2 = 2."""
    n = len(cartan)
    # a_ij * len_i^2 = a_ji * len_j^2
    lens = [None] * n  # type: ignore
    # Prefer a root that participates in an asymmetric bond as the short/long reference
    lens[0] = Fraction(2)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if lens[i] is None:
                continue
            for j in range(n):
                if i == j:
                    continue
                aij, aji = cartan[i][j], cartan[j][i]
                if aij == 0 and aji == 0:
                    continue
                # aij * li = aji * lj  => lj = aij * li / aji (if aji != 0)
                if aji != 0:
                    lj = Fraction(aij * lens[i], aji)
                    # aij is negative, aji negative, ratio positive
                    lj = abs(lj)  # lengths squared positive; signs of a cancel
                    # More carefully: aij*li = aji*lj, both a negative typically
                    lj = Fraction(aij, aji) * lens[i]
                    if lj <= 0:
                        # if one side zero bond asymmetric — shouldn't happen
                        continue
                    if lens[j] is None:
                        lens[j] = lj
                        changed = True
                    elif lens[j] != lj:
                        raise ValueError("Inconsistent Cartan length relations")
                elif aij != 0 and lens[j] is not None:
                    # aji=0 but aij!=0 impossible for symmetrizable
                    pass
    # Fill any disconnected (shouldn't happen) 
    for i in range(n):
        if lens[i] is None:
            lens[i] = Fraction(2)
    # Renormalize so max length^2 is 2 (long roots)
    max_len = max(lens)
    scale = Fraction(2) / max_len
    return tuple(li * scale for li in lens)


def _gram_matrix(cartan: Matrix) -> Tuple[Tuple[Fraction, ...], ...]:
    """Gram matrix G_ij = (α_i, α_j) = a_ij * (α_i,α_i) / 2."""
    n = len(cartan)
    lens = _simple_root_lengths_squared(cartan)
    G = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(Fraction(cartan[i][j]) * lens[i] / 2)
        G.append(tuple(row))
    return tuple(G)


def _coroot_coords_of_root(cartan: Matrix, root: Vector) -> Vector:
    """Express α∨ for a positive root α in the simple-coroot basis."""
    n = len(cartan)
    G = _gram_matrix(cartan)
    # (alpha, alpha)
    aa = Fraction(0)
    for i in range(n):
        for j in range(n):
            aa += root[i] * G[i][j] * root[j]
    # (alpha, alpha_j)
    alpha_dot_simples = []
    for j in range(n):
        val = Fraction(0)
        for i in range(n):
            val += root[i] * G[i][j]
        alpha_dot_simples.append(val)
    # <alpha^vee, alpha_j> = 2 (alpha, alpha_j) / (alpha, alpha)
    pairings = [2 * d / aa for d in alpha_dot_simples]
    # Solve A^T m = pairings, i.e. m^T A = pairings^T, pairing(m, e_j)=(A^T m)_j
    # Gaussian elimination over Q
    A = [[Fraction(cartan[i][j]) for j in range(n)] for i in range(n)]
    # Solve A^T m = v
    AT = [[A[j][i] for j in range(n)] for i in range(n)]
    aug = [AT[i] + [pairings[i]] for i in range(n)]
    for col in range(n):
        pivot = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for k in range(n + 1):
            aug[col][k] /= piv
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for k in range(n + 1):
                aug[r][k] -= factor * aug[col][k]
    m = []
    for i in range(n):
        val = aug[i][n]
        if val.denominator != 1:
            raise ValueError(f"Non-integral coroot coords {val} for root {root}")
        m.append(int(val))
    return tuple(m)


@dataclass(frozen=True)
class FiniteRootSystem:
    """Irreducible finite crystallographic root system from its Cartan matrix."""

    series: str
    rank: int
    cartan: Matrix

    @classmethod
    def create(cls, series: str, n: int) -> "FiniteRootSystem":
        series = series.upper()
        cartan = _finite_cartan(series, n)
        return cls(series=series, rank=n, cartan=cartan)

    @cached_property
    def positive_roots(self) -> Tuple[Vector, ...]:
        return _enumerate_positive_roots(self.cartan)

    @cached_property
    def positive_coroots(self) -> Tuple[Vector, ...]:
        return _enumerate_positive_roots(transpose(self.cartan))

    @cached_property
    def highest_root(self) -> Vector:
        return max(self.positive_roots, key=_height)

    @cached_property
    def highest_coroot(self) -> Vector:
        """θ∨ in simple-coroot coordinates (coroot of the highest root)."""
        return _coroot_coords_of_root(self.cartan, self.highest_root)

    @cached_property
    def coxeter_matrix(self) -> Matrix:
        return coxeter_from_cartan(self.cartan)

    @cached_property
    def simple_reflections(self) -> Tuple[Matrix, ...]:
        return tuple(reflection_on_coroots(self.cartan, i) for i in range(self.rank))

    @cached_property
    def simple_reflections_on_roots(self) -> Tuple[Matrix, ...]:
        return tuple(reflection_on_roots(self.cartan, i) for i in range(self.rank))

    def pairing(self, coroot_coords: Vector, root_coords: Vector) -> int:
        """⟨λ, α⟩ for λ in Q∨ (simple-coroot coords) and α (simple-root coords)."""
        Ay = _matmul(self.cartan, root_coords)
        return _dot(coroot_coords, Ay)

    def coroot_form(self, u: Vector, v: Vector) -> int:
        """Normalized invariant form ``(u|v)`` on the coroot lattice Q∨.

        With simple-coroot coordinates, ``(α_i∨|α_j∨) = a_{ji} (α_i∨|α_i∨)/2``
        where ``(α_i∨|α_i∨) = 4/(α_i|α_i)``.  Used for the linear action of
        translations on the affine coroot lattice.
        """
        if len(u) != self.rank or len(v) != self.rank:
            raise ValueError(
                f"coroot vectors must have length {self.rank}, got {len(u)}, {len(v)}"
            )
        root_lens = _simple_root_lengths_squared(self.cartan)
        coroot_lens = tuple(Fraction(4) / li for li in root_lens)
        n = self.rank
        val = Fraction(0)
        for i in range(n):
            for j in range(n):
                # G∨_ij = a_ji * |α_i∨|^2 / 2
                Gij = Fraction(self.cartan[j][i]) * coroot_lens[i] / 2
                val += u[i] * Gij * v[j]
        if val.denominator != 1:
            raise ValueError(f"Non-integral coroot form value {val}")
        return int(val)

    def is_positive_root(self, root_coords: Vector) -> bool:
        return root_coords in self._pos_set

    def is_negative_root(self, root_coords: Vector) -> bool:
        return _vec_scale(root_coords, -1) in self._pos_set

    @cached_property
    def _pos_set(self) -> frozenset:
        return frozenset(self.positive_roots)

    def finite_length(self, w_on_roots: Matrix) -> int:
        count = 0
        for alpha in self.positive_roots:
            image = _matmul(w_on_roots, alpha)
            if _height(image) < 0:
                count += 1
        return count

    @cached_property
    def weight_lattice(self) -> "WeightLattice":
        """Classical weight lattice ``P`` in the fundamental-weight basis."""
        from .weight_lattice import WeightLattice

        return WeightLattice(root_system=self)

    def fundamental_weight(self, i: int) -> Vector:
        """Fundamental weight ``ω_i`` (0-based) in fund-weight coordinates."""
        return self.weight_lattice.fundamental_weight(i)

    def pairing_weight_coroot(self, lam: Sequence[int], j: int) -> int:
        """``⟨λ, α_j∨⟩`` for ``λ ∈ P`` in fundamental-weight coordinates."""
        return self.weight_lattice.pairing_weight_coroot(lam, j)


def matrix_from_word_on_roots(rs: FiniteRootSystem, word: Iterable[int]) -> Matrix:
    M = _identity(rs.rank)
    for i in word:
        M = _matmatmul(rs.simple_reflections_on_roots[i], M)
    return M


def matrix_from_word_on_coroots(rs: FiniteRootSystem, word: Iterable[int]) -> Matrix:
    M = _identity(rs.rank)
    for i in word:
        M = _matmatmul(rs.simple_reflections[i], M)
    return M


__all__ = [
    "Vector",
    "FiniteRootSystem",
    "reflection_on_roots",
    "reflection_on_coroots",
    "_matmul",
    "_matmatmul",
    "_identity",
    "_vec_add",
    "_vec_sub",
    "_vec_scale",
    "_dot",
    "_height",
    "matrix_from_word_on_roots",
    "matrix_from_word_on_coroots",
]
