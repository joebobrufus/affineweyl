"""Affine Weyl group elements via the semidirect product W ⋉ Q∨.

Representation
--------------
An element is stored as a pair ``(w, λ)`` meaning the affine isometry

    x ↦ w(x) + λ

i.e. the product ``t_λ ∘ w`` of a finite Weyl element ``w ∈ W`` and a
translation by a coroot-lattice vector ``λ ∈ Q∨``.

Multiplication (compose right-to-left as functions)::

    (w, λ) * (w', λ') = (w w', λ + w(λ'))

Inversion::

    (w, λ)^{-1} = (w^{-1}, -w^{-1}(λ))

Length (Iwahori–Matsumoto)::

    ℓ(t_λ w) = Σ_{α ∈ Φ+} |⟨λ, α⟩ - χ(w^{-1}·α ∈ Φ-)|

Equality is by the pair ``(w, λ)`` (canonical normal forms), not word equality.

Each element also carries its linear action on the affine root and coroot
lattices (coordinates in the affine simple (co)root bases of size n+1).
If ``α = x · α_i`` for a simple root ``α_i``, the associated coroot is
``α∨ = x · α_i∨`` — same ``x``, corresponding simple coroot.

Projection to the finite Weyl group
---------------------------------
The quotient map ``π: W̃ → W̃ / Q∨ ≅ W`` drops the translation and keeps
``w``: ``π(w, λ) = w``.  Use :meth:`AffineWeylElement.to_finite` /
:attr:`finite_part`.  Translations form the kernel.

No I/O in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, List, Sequence, Tuple

from .root_system import (
    Matrix,
    Vector,
    _height,
    _identity,
    _matmul,
    _matmatmul,
    _vec_add,
    _vec_scale,
)

if TYPE_CHECKING:
    from .group import AffineWeylGroup


def _invert_w_matrix(M: Matrix) -> Matrix:
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
class AffineWeylElement:
    """Immutable, hashable element of an untwisted affine Weyl group.

    Attributes
    ----------
    group :
        Parent :class:`~affineweyl.group.AffineWeylGroup`.
    w_coroots, w_roots :
        Matrices of the finite part ``w`` on simple (co)root coordinates.
    translation :
        Coroot-lattice vector ``λ`` in simple-coroot coordinates.
    aff_roots, aff_coroots :
        Matrices of the full affine action on affine simple (co)root
        coordinates (size ``affine_rank × affine_rank``).
    """

    group: "AffineWeylGroup"
    w_coroots: Matrix
    w_roots: Matrix
    translation: Vector
    aff_roots: Matrix
    aff_coroots: Matrix

    @staticmethod
    def identity(group: "AffineWeylGroup") -> "AffineWeylElement":
        n = group.finite_rank
        a = group.affine_rank
        return AffineWeylElement(
            group=group,
            w_coroots=_identity(n),
            w_roots=_identity(n),
            translation=tuple(0 for _ in range(n)),
            aff_roots=_identity(a),
            aff_coroots=_identity(a),
        )

    @staticmethod
    def simple(group: "AffineWeylGroup", i: int) -> "AffineWeylElement":
        """Return the simple generator ``s_i`` (``i`` in ``0..n``)."""
        if i < 0 or i > group.finite_rank:
            raise IndexError(f"simple index {i} out of range 0..{group.finite_rank}")
        if i == 0:
            return group._make_s0()
        j = i - 1
        a = group.affine_rank
        return AffineWeylElement(
            group=group,
            w_coroots=group.root_system.simple_reflections[j],
            w_roots=group.root_system.simple_reflections_on_roots[j],
            translation=tuple(0 for _ in range(group.finite_rank)),
            aff_roots=group.affine_simple_on_roots[i],
            aff_coroots=group.affine_simple_on_coroots[i],
        )

    def __mul__(self, other: "AffineWeylElement") -> "AffineWeylElement":
        if not isinstance(other, AffineWeylElement):
            return NotImplemented
        if self.group.label != other.group.label:
            raise ValueError("Cannot multiply elements of different affine Weyl groups")
        w_c = _matmatmul(self.w_coroots, other.w_coroots)
        w_r = _matmatmul(self.w_roots, other.w_roots)
        w_lambda = _matmul(self.w_coroots, other.translation)
        lam = _vec_add(self.translation, w_lambda)
        return AffineWeylElement(
            group=self.group,
            w_coroots=w_c,
            w_roots=w_r,
            translation=lam,
            aff_roots=_matmatmul(self.aff_roots, other.aff_roots),
            aff_coroots=_matmatmul(self.aff_coroots, other.aff_coroots),
        )

    def __pow__(self, n: int) -> "AffineWeylElement":
        if n == 0:
            return AffineWeylElement.identity(self.group)
        if n < 0:
            return (self.inv()) ** (-n)
        result = AffineWeylElement.identity(self.group)
        base = self
        exp = n
        while exp:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        return result

    def inv(self) -> "AffineWeylElement":
        """Inverse: ``(w, λ)^{-1} = (w^{-1}, -w^{-1}(λ))``."""
        w_c_inv = _invert_w_matrix(self.w_coroots)
        w_r_inv = _invert_w_matrix(self.w_roots)
        lam = _vec_scale(_matmul(w_c_inv, self.translation), -1)
        return AffineWeylElement(
            group=self.group,
            w_coroots=w_c_inv,
            w_roots=w_r_inv,
            translation=lam,
            aff_roots=_invert_w_matrix(self.aff_roots),
            aff_coroots=_invert_w_matrix(self.aff_coroots),
        )

    def __invert__(self) -> "AffineWeylElement":
        return self.inv()

    def left_multiply_simple(self, i: int) -> "AffineWeylElement":
        return AffineWeylElement.simple(self.group, i) * self

    def right_multiply_simple(self, i: int) -> "AffineWeylElement":
        return self * AffineWeylElement.simple(self.group, i)

    # --- action on affine roots / coroots ---------------------------------

    def act_on_root(self, root: Sequence[int]) -> Vector:
        """Act on a vector in affine simple-root coordinates."""
        if len(root) != self.group.affine_rank:
            raise ValueError(
                f"root has length {len(root)}, expected {self.group.affine_rank}"
            )
        return _matmul(self.aff_roots, tuple(int(x) for x in root))

    def act_on_coroot(self, coroot: Sequence[int]) -> Vector:
        """Act on a vector in affine simple-coroot coordinates."""
        if len(coroot) != self.group.affine_rank:
            raise ValueError(
                f"coroot has length {len(coroot)}, expected {self.group.affine_rank}"
            )
        return _matmul(self.aff_coroots, tuple(int(x) for x in coroot))

    def image_of_simple_root(self, i: int) -> Vector:
        """Return ``x · α_i`` in affine simple-root coordinates."""
        return self.act_on_root(self.group.simple_root(i))

    def image_of_simple_coroot(self, i: int) -> Vector:
        """Return ``x · α_i∨`` in affine simple-coroot coordinates."""
        return self.act_on_coroot(self.group.simple_coroot(i))

    def associated_coroot(self, i: int) -> Vector:
        """If ``α = x · α_i``, return the associated coroot ``α∨ = x · α_i∨``.

        This is the defining association between real affine roots and coroots
        obtained from a simple root via the affine Weyl group.
        """
        return self.image_of_simple_coroot(i)

    def root_coroot_pair(self, i: int) -> Tuple[Vector, Vector]:
        """Return ``(x · α_i, x · α_i∨)``."""
        return self.image_of_simple_root(i), self.image_of_simple_coroot(i)

    # --- length / words ---------------------------------------------------

    @cached_property
    def length(self) -> int:
        """Coxeter length via the Iwahori–Matsumoto formula."""
        rs = self.group.root_system
        w_inv = _invert_w_matrix(self.w_roots)
        total = 0
        for alpha in rs.positive_roots:
            pair = rs.pairing(self.translation, alpha)
            image = _matmul(w_inv, alpha)
            chi = 1 if _height(image) < 0 else 0
            total += abs(pair - chi)
        return total

    def is_identity(self) -> bool:
        n = self.group.finite_rank
        if any(x != 0 for x in self.translation):
            return False
        I = _identity(n)
        return self.w_roots == I and self.w_coroots == I

    def reduced_word(self) -> Tuple[int, ...]:
        """A reduced expression as a tuple of simple indices (greedy right descent)."""
        if self.is_identity():
            return ()
        word_rev: List[int] = []
        x = self
        rank = self.group.affine_rank
        seen_max = x.length + 1
        while not x.is_identity():
            ell = x.length
            if ell >= seen_max or ell < 0:
                raise RuntimeError("Length did not decrease while building reduced word")
            seen_max = ell
            found = False
            for i in range(rank):
                y = x.right_multiply_simple(i)
                if y.length < ell:
                    word_rev.append(i)
                    x = y
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"No right descent found at length {ell} for element of {self.group.label}"
                )
        return tuple(reversed(word_rev))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineWeylElement):
            return NotImplemented
        return (
            self.group.label == other.group.label
            and self.w_roots == other.w_roots
            and self.w_coroots == other.w_coroots
            and self.translation == other.translation
        )

    def __hash__(self) -> int:
        return hash((self.group.label, self.w_roots, self.w_coroots, self.translation))


    # --- projection π: W̃ → W ---------------------------------------------

    def to_finite(self) -> "FiniteWeylElement":
        """Project to the finite Weyl group via ``π(w, λ) = w``.

        This is the quotient map ``W̃ → W̃ / Q∨ ≅ W``: translations
        ``t_λ`` form the kernel, and ``π`` is the identity on pure finite
        elements (``translation = 0``).  Homomorphism: ``π(xy) = π(x)π(y)``.
        """
        from .finite import FiniteWeylElement

        return FiniteWeylElement(
            group=self.group,
            w_roots=self.w_roots,
            w_coroots=self.w_coroots,
        )

    @property
    def finite_part(self) -> "FiniteWeylElement":
        """Alias for :meth:`to_finite` (the finite Weyl factor ``w``)."""
        return self.to_finite()

    def __repr__(self) -> str:
        if self.is_identity():
            return f"id({self.group.label})"
        word = self.reduced_word()
        return f"AffineWeylElement({self.group.label}, word={word}, length={self.length})"

    def __str__(self) -> str:
        word = self.reduced_word()
        if not word:
            return "1"
        return " ".join(f"s{i}" for i in word)


__all__ = ["AffineWeylElement"]
