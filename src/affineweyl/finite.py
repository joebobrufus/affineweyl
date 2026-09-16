"""Finite Weyl group elements: image of the quotient map π: W̃ → W.

The natural projection

    π: W̃ ≅ W ⋉ Q∨  →  W ≅ W̃ / Q∨

sends ``(w, λ) ↦ w``.  Equivalently, translations ``t_λ = (id, λ)`` form the
kernel, and π restricts to the identity on the embedded copy of the finite
Weyl group (elements with ``translation = 0``).

Homomorphism property: ``π(xy) = π(x) π(y)``.

Finite elements act on the classical weight lattice ``P`` (fundamental-weight
coordinates) via :meth:`FiniteWeylElement.act_on_weight`.

No I/O in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Sequence, Tuple

from .root_system import Matrix, Vector, _identity, _matmul, _matmatmul

if TYPE_CHECKING:
    from .group import AffineWeylGroup
    from .weight_lattice import WeightCoords


def _invert_matrix(M: Matrix) -> Matrix:
    """Invert a matrix in GL(n, Z) via Gaussian elimination over rationals."""
    n = len(M)
    aug = [
        [Fraction(M[i][j]) for j in range(n)]
        + [Fraction(1 if i == j else 0) for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Singular matrix — not a Weyl element")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for k in range(2 * n):
            aug[col][k] /= piv
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for k in range(2 * n):
                aug[r][k] -= factor * aug[col][k]
    inv_rows = []
    for i in range(n):
        row = []
        for j in range(n):
            val = aug[i][n + j]
            if val.denominator != 1:
                raise ValueError("Inverse not integral")
            row.append(int(val))
        inv_rows.append(tuple(row))
    return tuple(inv_rows)


@dataclass(frozen=True)
class FiniteWeylElement:
    """Immutable element of the finite Weyl group ``W ≅ W̃ / Q∨``.

    Obtained as the image of an affine element under the quotient map
    ``π: W̃ → W`` that drops the translation and keeps the finite part
    ``(w_roots, w_coroots)``.

    Attributes
    ----------
    group :
        Parent :class:`~affineweyl.group.AffineWeylGroup` (same Cartan type).
    w_roots, w_coroots :
        Matrices of ``w ∈ W`` on simple root / coroot coordinates.
    """

    group: "AffineWeylGroup"
    w_roots: Matrix
    w_coroots: Matrix

    @staticmethod
    def identity(group: "AffineWeylGroup") -> "FiniteWeylElement":
        n = group.finite_rank
        return FiniteWeylElement(
            group=group,
            w_roots=_identity(n),
            w_coroots=_identity(n),
        )

    @staticmethod
    def simple(group: "AffineWeylGroup", i: int) -> "FiniteWeylElement":
        """Finite simple reflection ``s_i`` with ``i`` in ``0 .. n-1``."""
        n = group.finite_rank
        if i < 0 or i >= n:
            raise IndexError(f"finite simple index {i} out of range 0..{n - 1}")
        rs = group.root_system
        return FiniteWeylElement(
            group=group,
            w_roots=rs.simple_reflections_on_roots[i],
            w_coroots=rs.simple_reflections[i],
        )

    def __mul__(self, other: "FiniteWeylElement") -> "FiniteWeylElement":
        if not isinstance(other, FiniteWeylElement):
            return NotImplemented
        if self.group.label != other.group.label:
            raise ValueError("Cannot multiply elements of different finite Weyl groups")
        return FiniteWeylElement(
            group=self.group,
            w_roots=_matmatmul(self.w_roots, other.w_roots),
            w_coroots=_matmatmul(self.w_coroots, other.w_coroots),
        )

    def inv(self) -> "FiniteWeylElement":
        return FiniteWeylElement(
            group=self.group,
            w_roots=_invert_matrix(self.w_roots),
            w_coroots=_invert_matrix(self.w_coroots),
        )

    def __invert__(self) -> "FiniteWeylElement":
        return self.inv()

    def __pow__(self, n: int) -> "FiniteWeylElement":
        if n == 0:
            return FiniteWeylElement.identity(self.group)
        if n < 0:
            return (self.inv()) ** (-n)
        result = FiniteWeylElement.identity(self.group)
        base = self
        exp = n
        while exp:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        return result

    def is_identity(self) -> bool:
        I = _identity(self.group.finite_rank)
        return self.w_roots == I and self.w_coroots == I

    def act_on_root(self, root: Sequence[int]) -> Vector:
        """Act on a vector in finite simple-root coordinates."""
        if len(root) != self.group.finite_rank:
            raise ValueError(
                f"root has length {len(root)}, expected {self.group.finite_rank}"
            )
        return _matmul(self.w_roots, tuple(int(x) for x in root))

    def act_on_coroot(self, coroot: Sequence[int]) -> Vector:
        """Act on a vector in finite simple-coroot coordinates."""
        if len(coroot) != self.group.finite_rank:
            raise ValueError(
                f"coroot has length {len(coroot)}, expected {self.group.finite_rank}"
            )
        return _matmul(self.w_coroots, tuple(int(x) for x in coroot))

    def act_on_weight(self, lam: Sequence[int]) -> "WeightCoords":
        """Act on a weight in fundamental-weight coordinates.

        Implements the linear action of ``w ∈ W`` on the weight lattice ``P``.
        Coordinates are converted to the simple-root basis (over ``Q``),
        transformed by :attr:`w_roots`, then converted back with the Cartan
        matrix — consistent with the existing root action and
        :class:`~affineweyl.weight_lattice.WeightLattice` converters.

        Equivalently, for a simple reflection,
        ``s_i(λ) = λ - ⟨λ, α_i∨⟩ α_i``, and since ``⟨λ, α_i∨⟩`` is the ``i``-th
        fund-weight coordinate of ``λ`` while ``α_i`` has fund-weight
        coordinates equal to the ``i``-th column of the Cartan matrix ``A``,
        one has ``s_i(λ)_fund = λ - λ_i · (column i of A)``.

        Parameters
        ----------
        lam :
            Weight as an ``n``-tuple of integers in the ``ω``-basis.

        Returns
        -------
        WeightCoords
            ``w · λ`` in the same fundamental-weight coordinates.
        """
        n = self.group.finite_rank
        if len(lam) != n:
            raise ValueError(f"weight has length {len(lam)}, expected {n}")
        for i, x in enumerate(lam):
            if isinstance(x, bool) or not isinstance(x, int):
                raise TypeError(
                    f"weight coordinate {i} must be an int, got {type(x).__name__}"
                )
        P = self.group.weight_lattice
        # μ = A^{-1} λ  (simple-root coords over Q), then A (w_roots μ)
        root_coords = P.to_simple_root_coords(lam)
        acted = tuple(
            sum(Fraction(self.w_roots[i][j]) * root_coords[j] for j in range(n))
            for i in range(n)
        )
        A = P.cartan
        out = []
        for i in range(n):
            val = sum(Fraction(A[i][j]) * acted[j] for j in range(n))
            if val.denominator != 1:
                raise ValueError(f"Non-integral weight coordinate {val}")
            out.append(int(val))
        return tuple(out)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FiniteWeylElement):
            return NotImplemented
        return (
            self.group.label == other.group.label
            and self.w_roots == other.w_roots
            and self.w_coroots == other.w_coroots
        )

    def __hash__(self) -> int:
        return hash((self.group.label, self.w_roots, self.w_coroots))

    def __repr__(self) -> str:
        if self.is_identity():
            return f"id_W({self.group.label})"
        return f"FiniteWeylElement({self.group.label}, w_roots={self.w_roots})"


__all__ = ["FiniteWeylElement"]
