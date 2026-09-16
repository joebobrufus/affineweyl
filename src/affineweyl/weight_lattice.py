"""Classical (finite) weight lattice P in the fundamental-weight basis.

Weights are length-``n`` integer tuples ``(λ_0, …, λ_{n-1})`` meaning
``λ = Σ_i λ_i ω_i``, where ``ω_i`` are the fundamental weights defined by
``⟨ω_i, α_j∨⟩ = δ_{ij}``.  Simple-root indices are **0-based** (``0 .. n-1``),
matching :class:`~affineweyl.root_system.FiniteRootSystem`.

Conversion to simple-root coordinates uses the Cartan matrix ``A``:
``α_j = Σ_i a_{ij} ω_i`` in this package's convention
``a_{ij} = ⟨α_i∨, α_j⟩``, so a root with simple-root coords ``μ`` has
fundamental-weight coords ``A μ``.  Conversely, fund-weight coords ``λ``
correspond to simple-root coords ``A^{-1} λ`` (rational in general).

The root lattice satisfies ``Q ⊂ P`` with index ``|P/Q| = |det A|``.

The finite Weyl group acts on ``P``; see :meth:`WeightLattice.act` and
:meth:`~affineweyl.finite.FiniteWeylElement.act_on_weight`.

The symmetric algebra ``Sym(P) ≅ ℤ[X_0,…,X_{n-1}]`` (fundamental-weight
variables) is :attr:`symmetric_algebra`; see :mod:`affineweyl.symmetric_algebra`.

No I/O in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Sequence, Tuple, Union

from .cartan import Matrix
from .root_system import FiniteRootSystem, Vector, _dot, _matmul

if TYPE_CHECKING:
    from .element import AffineWeylElement
    from .finite import FiniteWeylElement
    from .symmetric_algebra import SymmetricAlgebra

WeightCoords = Tuple[int, ...]  # fundamental-weight basis
RationalCoords = Tuple[Fraction, ...]


def _as_int_tuple(coords: Sequence[int], n: int, name: str) -> WeightCoords:
    if len(coords) != n:
        raise ValueError(f"{name} has length {len(coords)}, expected {n}")
    out = []
    for i, x in enumerate(coords):
        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError(
                f"{name} coordinate {i} must be an int, got {type(x).__name__}"
            )
        out.append(x)
    return tuple(out)


def _matrix_det(M: Matrix) -> int:
    """Determinant of an integer matrix (exact, via Fraction elimination)."""
    n = len(M)
    A = [[Fraction(M[i][j]) for j in range(n)] for i in range(n)]
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if A[r][col] != 0), None)
        if pivot is None:
            return 0
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            det = -det
        det *= A[col][col]
        piv = A[col][col]
        for r in range(col + 1, n):
            factor = A[r][col] / piv
            for k in range(col, n):
                A[r][k] -= factor * A[col][k]
    if det.denominator != 1:
        raise ValueError(f"Non-integral determinant {det}")
    return int(det)


def _solve_cartan(A: Matrix, b: Sequence[int]) -> RationalCoords:
    """Solve ``A x = b`` over ``Q`` by Gaussian elimination."""
    n = len(A)
    aug = [
        [Fraction(A[i][j]) for j in range(n)] + [Fraction(b[i])]
        for i in range(n)
    ]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("Singular Cartan matrix")
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
    return tuple(aug[i][n] for i in range(n))


@dataclass(frozen=True)
class WeightLattice:
    """Weight lattice ``P`` of an irreducible finite root system.

    Elements are integer tuples in the **fundamental-weight basis**:
    ``(λ_0, …, λ_{n-1})`` means ``λ = Σ λ_i ω_i``.
    """

    root_system: FiniteRootSystem

    @property
    def rank(self) -> int:
        return self.root_system.rank

    @property
    def series(self) -> str:
        return self.root_system.series

    @property
    def cartan(self) -> Matrix:
        return self.root_system.cartan

    def fundamental_weight(self, i: int) -> WeightCoords:
        """Return ``ω_i`` in fundamental-weight coordinates (0-based index)."""
        n = self.rank
        if i < 0 or i >= n:
            raise IndexError(i)
        return tuple(1 if j == i else 0 for j in range(n))

    def pairing_weight_coroot(self, lam: Sequence[int], j: int) -> int:
        """Pairing ``⟨λ, α_j∨⟩`` for ``λ ∈ P`` (fund-weight coords).

        Equals the ``j``-th fundamental-weight coordinate of ``λ``, so
        ``⟨ω_i, α_j∨⟩ = δ_{ij}``.
        """
        coords = _as_int_tuple(lam, self.rank, "weight")
        if j < 0 or j >= self.rank:
            raise IndexError(j)
        return coords[j]

    def from_simple_root_coords(self, root_coords: Sequence[int]) -> WeightCoords:
        """Convert simple-root coordinates to fundamental-weight coordinates.

        If ``μ = Σ μ_i α_i``, the fund-weight coords are ``A μ``.
        """
        mu = _as_int_tuple(root_coords, self.rank, "root coordinates")
        return _matmul(self.cartan, mu)

    def to_simple_root_coords(self, lam: Sequence[int]) -> RationalCoords:
        """Convert fund-weight coordinates to simple-root coordinates over ``Q``.

        Solves ``A x = λ``.  The result is integral iff ``λ ∈ Q``.
        """
        coords = _as_int_tuple(lam, self.rank, "weight")
        return _solve_cartan(self.cartan, coords)

    def is_in_root_lattice(self, lam: Sequence[int]) -> bool:
        """Return True iff ``λ ∈ Q`` (simple-root coords of ``λ`` are integral)."""
        x = self.to_simple_root_coords(lam)
        return all(c.denominator == 1 for c in x)

    @cached_property
    def index_P_mod_Q(self) -> int:
        """Index ``|P/Q| = |det A|`` of the root lattice in the weight lattice."""
        return abs(_matrix_det(self.cartan))

    # --- optional coweight lattice helpers (dual picture) ---

    def fundamental_coweight(self, i: int) -> WeightCoords:
        """Return ``ω_i∨`` in fundamental-coweight coordinates (standard basis).

        Coweights use the same coordinate convention as weights: an integer
        tuple ``(ν_0, …, ν_{n-1})`` means ``ν = Σ ν_i ω_i∨``, with
        ``⟨α_j, ω_i∨⟩ = δ_{ji}``.
        """
        return self.fundamental_weight(i)

    def pairing_root_coweight(self, coweight: Sequence[int], j: int) -> int:
        """Pairing ``⟨α_j, ν⟩`` for ``ν ∈ P∨`` in fund-coweight coords.

        Equals the ``j``-th coordinate of ``ν``.
        """
        return self.pairing_weight_coroot(coweight, j)

    def from_simple_coroot_coords(self, coroot_coords: Sequence[int]) -> WeightCoords:
        """Convert simple-coroot coords to fundamental-coweight coords (``A μ``)."""
        return self.from_simple_root_coords(coroot_coords)

    def is_in_coroot_lattice(self, coweight: Sequence[int]) -> bool:
        """Return True iff the coweight lies in the coroot lattice ``Q∨``."""
        return self.is_in_root_lattice(coweight)

    def act(
        self,
        w: "Union[FiniteWeylElement, AffineWeylElement]",
        lam: Sequence[int],
    ) -> WeightCoords:
        """Return ``w · λ`` in fundamental-weight coordinates.

        ``w`` may be a :class:`~affineweyl.finite.FiniteWeylElement` or an
        :class:`~affineweyl.element.AffineWeylElement` (the latter acts via
        its finite projection ``π``, so translations act as the identity on
        ``P``).

        See :meth:`~affineweyl.finite.FiniteWeylElement.act_on_weight` for the
        coordinate convention (``w_roots`` on simple-root coords, converted
        back with the Cartan matrix).
        """
        from .element import AffineWeylElement
        from .finite import FiniteWeylElement

        if isinstance(w, AffineWeylElement):
            w = w.to_finite()
        if not isinstance(w, FiniteWeylElement):
            raise TypeError(
                "w must be a FiniteWeylElement or AffineWeylElement, "
                f"got {type(w).__name__}"
            )
        if w.group.series != self.series or w.group.finite_rank != self.rank:
            raise ValueError(
                f"element type {w.group.label} incompatible with "
                f"WeightLattice({self.series}_{self.rank})"
            )
        return w.act_on_weight(lam)


    @cached_property
    def symmetric_algebra(self) -> "SymmetricAlgebra":
        """Polynomial ring ``Sym(P) ≅ ℤ[X_0, …, X_{n-1}]`` on fundamental weights.

        See :class:`~affineweyl.symmetric_algebra.SymmetricAlgebra`.
        """
        from .symmetric_algebra import SymmetricAlgebra

        return SymmetricAlgebra(weight_lattice=self)

    def __repr__(self) -> str:
        return f"WeightLattice({self.series}_{self.rank})"


def weight_lattice(rs: FiniteRootSystem) -> WeightLattice:
    """Return the weight lattice of a finite root system."""
    return WeightLattice(root_system=rs)


__all__ = [
    "WeightCoords",
    "RationalCoords",
    "WeightLattice",
    "weight_lattice",
]
