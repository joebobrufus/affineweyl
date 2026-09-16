"""Affine Weyl groups (untwisted) as Coxeter groups realized by W ⋉ Q∨.

The natural projection ``π: W̃ → W ≅ W̃ / Q∨`` (drop translation) is exposed
as :meth:`AffineWeylGroup.project_to_finite` / :meth:`finite_projection`.

No I/O in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List, Sequence, Set, Tuple

from .cartan import (
    Matrix,
    canonical_label,
    coxeter_from_cartan,
    parse_affine_type,
    validate_rank,
)
from .element import AffineWeylElement
from .finite import FiniteWeylElement
from .root_system import (
    FiniteRootSystem,
    Vector,
    _identity,
    reflection_on_coroots,
    reflection_on_roots,
)


def _affine_cartan(finite: Matrix, highest_root: Vector, highest_coroot: Vector) -> Matrix:
    """Build the untwisted affine Cartan matrix of size (n+1) x (n+1).

    Indexing: row/col 0 = affine node; 1..n = finite nodes 0..n-1.
    a_{0,i+1} = -⟨θ∨, α_i⟩, a_{i+1,0} = -⟨α_i∨, θ⟩.
    """
    n = len(finite)
    a0 = []
    for i in range(n):
        val = sum(highest_coroot[r] * finite[r][i] for r in range(n))
        a0.append(-val)
    ai0 = []
    for i in range(n):
        val = sum(finite[i][c] * highest_root[c] for c in range(n))
        ai0.append(-val)

    rows: List[List[int]] = [[0] * (n + 1) for _ in range(n + 1)]
    rows[0][0] = 2
    for i in range(n):
        rows[0][i + 1] = a0[i]
        rows[i + 1][0] = ai0[i]
        for j in range(n):
            rows[i + 1][j + 1] = finite[i][j]
    return tuple(tuple(r) for r in rows)


@dataclass(frozen=True)
class AffineWeylGroup:
    """Untwisted affine Weyl group of a given type.

    Parameters
    ----------
    series :
        One of ``A,B,C,D,E,F,G``.
    n :
        Finite rank (affine rank is ``n+1``).
    """

    series: str
    n: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "series", self.series.upper())
        validate_rank(self.series, self.n)

    @classmethod
    def from_label(cls, label: str) -> "AffineWeylGroup":
        series, n = parse_affine_type(label)
        return cls(series=series, n=n)

    @property
    def label(self) -> str:
        return canonical_label(self.series, self.n)

    @property
    def finite_rank(self) -> int:
        return self.n

    @property
    def affine_rank(self) -> int:
        """Number of simple generators (n+1)."""
        return self.n + 1

    @cached_property
    def root_system(self) -> FiniteRootSystem:
        return FiniteRootSystem.create(self.series, self.n)

    @cached_property
    def finite_cartan(self) -> Matrix:
        return self.root_system.cartan

    @cached_property
    def cartan_matrix(self) -> Matrix:
        rs = self.root_system
        return _affine_cartan(rs.cartan, rs.highest_root, rs.highest_coroot)

    @cached_property
    def coxeter_matrix(self) -> Matrix:
        return coxeter_from_cartan(self.cartan_matrix)

    @cached_property
    def affine_simple_on_roots(self) -> Tuple[Matrix, ...]:
        """Reflection matrices of s_0..s_n on affine simple-root coordinates."""
        C = self.cartan_matrix
        return tuple(reflection_on_roots(C, i) for i in range(self.affine_rank))

    @cached_property
    def affine_simple_on_coroots(self) -> Tuple[Matrix, ...]:
        """Reflection matrices of s_0..s_n on affine simple-coroot coordinates."""
        C = self.cartan_matrix
        return tuple(reflection_on_coroots(C, i) for i in range(self.affine_rank))

    def simple_root(self, i: int) -> Vector:
        """Affine simple root ``α_i`` as a standard basis vector in Z^{n+1}."""
        if i < 0 or i >= self.affine_rank:
            raise IndexError(i)
        return tuple(1 if j == i else 0 for j in range(self.affine_rank))

    def simple_coroot(self, i: int) -> Vector:
        """Affine simple coroot ``α_i∨`` as a standard basis vector in Z^{n+1}."""
        if i < 0 or i >= self.affine_rank:
            raise IndexError(i)
        return tuple(1 if j == i else 0 for j in range(self.affine_rank))

    def _make_s0(self) -> AffineWeylElement:
        """Affine simple reflection s_0 = t_{θ∨} ∘ s_θ."""
        rs = self.root_system
        theta = rs.highest_root
        theta_vee = rs.highest_coroot
        n = self.n
        rows_r = []
        for j in range(n):
            ej = tuple(1 if k == j else 0 for k in range(n))
            coef = rs.pairing(theta_vee, ej)
            img = tuple(ej[k] - coef * theta[k] for k in range(n))
            rows_r.append(img)
        w_roots = tuple(tuple(rows_r[c][r] for c in range(n)) for r in range(n))

        rows_c = []
        for j in range(n):
            ej = tuple(1 if k == j else 0 for k in range(n))
            coef = sum(rs.cartan[j][c] * theta[c] for c in range(n))
            img = tuple(ej[k] - coef * theta_vee[k] for k in range(n))
            rows_c.append(img)
        w_coroots = tuple(tuple(rows_c[c][r] for c in range(n)) for r in range(n))

        return AffineWeylElement(
            group=self,
            w_coroots=w_coroots,
            w_roots=w_roots,
            translation=theta_vee,
            aff_roots=self.affine_simple_on_roots[0],
            aff_coroots=self.affine_simple_on_coroots[0],
        )

    @property
    def s0(self) -> AffineWeylElement:
        return self._make_s0()

    def _aff_matrices_of_translation(self, lam: Vector) -> Tuple[Matrix, Matrix]:
        """Affine (co)root matrices of the pure translation ``t_λ``.

        On roots (untwisted): ``t_λ(β + mδ) = β + (m - ⟨λ, β⟩)δ``.
        In affine simple-root coordinates, with ``α_0 = δ - θ``::

            c'_0 = c_0 - ⟨λ, β⟩,
            c'_i = c_i - ⟨λ, β⟩ θ_i   (i = 1..n),

        where ``β_i = c_i - c_0 θ_i`` is the finite root part.

        On coroots the same shape holds with ``θ∨`` and the invariant form
        ``(λ|γ)`` on Q∨ in place of ``⟨λ, β⟩``.
        """
        rs = self.root_system
        n = self.finite_rank
        a = self.affine_rank
        theta = rs.highest_root
        theta_vee = rs.highest_coroot

        roots_rows: List[List[int]] = [[0] * a for _ in range(a)]
        for j in range(a):
            c = [1 if k == j else 0 for k in range(a)]
            c0 = c[0]
            beta = tuple(c[i + 1] - c0 * theta[i] for i in range(n))
            pair = rs.pairing(lam, beta)
            roots_rows[0][j] = c0 - pair
            for i in range(n):
                roots_rows[i + 1][j] = c[i + 1] - pair * theta[i]

        coroots_rows: List[List[int]] = [[0] * a for _ in range(a)]
        for j in range(a):
            d = [1 if k == j else 0 for k in range(a)]
            d0 = d[0]
            gamma = tuple(d[i + 1] - d0 * theta_vee[i] for i in range(n))
            pair = rs.coroot_form(lam, gamma)
            coroots_rows[0][j] = d0 - pair
            for i in range(n):
                coroots_rows[i + 1][j] = d[i + 1] - pair * theta_vee[i]

        return (
            tuple(tuple(r) for r in roots_rows),
            tuple(tuple(r) for r in coroots_rows),
        )

    def translation(self, lam: Sequence[int]) -> AffineWeylElement:
        """Return the pure translation ``t_λ`` in ``W ⋉ Q∨``.

        Parameters
        ----------
        lam :
            Coefficients of ``λ ∈ Q∨`` in the finite simple-coroot basis
            ``α_1∨, …, α_n∨`` (length ``finite_rank``).

        Returns
        -------
        AffineWeylElement
            The element ``(id, λ)``, i.e. finite Weyl part = identity and
            ``translation = tuple(lam)``, with affine (co)root actions equal
            to the linear action of ``t_λ`` (not the identity matrices).

        Notes
        -----
        Consistent with ``s_0 = t_{θ∨} ∘ s_θ`` from :meth:`_make_s0`: one has
        ``W.simple(0) == W.translation(θ∨) * s_θ`` when ``s_θ`` is the finite
        reflection in the highest root (embedded via ``s_1…s_n``).
        Translations multiply by adding lattice vectors:
        ``t_λ * t_μ = t_{λ+μ}``.
        """
        n = self.finite_rank
        if len(lam) != n:
            raise ValueError(
                f"translation vector has length {len(lam)}, expected {n}"
            )
        coords: List[int] = []
        for i, x in enumerate(lam):
            if isinstance(x, bool) or not isinstance(x, int):
                raise TypeError(
                    f"translation coordinate {i} must be an int, got {type(x).__name__}"
                )
            coords.append(x)
        lam_t: Vector = tuple(coords)
        if all(x == 0 for x in lam_t):
            return self.identity()
        aff_roots, aff_coroots = self._aff_matrices_of_translation(lam_t)
        return AffineWeylElement(
            group=self,
            w_coroots=_identity(n),
            w_roots=_identity(n),
            translation=lam_t,
            aff_roots=aff_roots,
            aff_coroots=aff_coroots,
        )

    def identity(self) -> AffineWeylElement:
        return AffineWeylElement.identity(self)

    def simple(self, i: int) -> AffineWeylElement:
        return AffineWeylElement.simple(self, i)

    def generators(self) -> Tuple[AffineWeylElement, ...]:
        return tuple(self.simple(i) for i in range(self.affine_rank))

    def from_word(self, word: Sequence[int]) -> AffineWeylElement:
        """Product ``s_{i1} ... s_{ik}`` (left-to-right multiplication)."""
        x = self.identity()
        for i in word:
            x = x * self.simple(int(i))
        return x

    def is_reduced(self, word: Sequence[int]) -> bool:
        """Return True iff ``word`` is a reduced expression."""
        return self.from_word(word).length == len(word)

    def m(self, i: int, j: int) -> int:
        """Coxeter integer m_{ij} (0 means infinity)."""
        return self.coxeter_matrix[i][j]

    def elements_up_to_length(self, max_length: int) -> List[AffineWeylElement]:
        """Enumerate all elements of length ``<= max_length`` (BFS)."""
        if max_length < 0:
            return []
        identity = self.identity()
        result: List[AffineWeylElement] = [identity]
        by_len: List[List[AffineWeylElement]] = [[identity]]
        seen: Set[AffineWeylElement] = {identity}
        for ell in range(max_length):
            nxt: List[AffineWeylElement] = []
            for x in by_len[ell]:
                for i in range(self.affine_rank):
                    y = x.right_multiply_simple(i)
                    if y.length == ell + 1 and y not in seen:
                        seen.add(y)
                        nxt.append(y)
                        result.append(y)
            by_len.append(nxt)
            if not nxt:
                break
        return result

    @cached_property
    def weight_lattice(self):
        """Classical weight lattice ``P`` of the underlying finite root system."""
        return self.root_system.weight_lattice

    def fundamental_weight(self, i: int) -> Vector:
        """Fundamental weight ``ω_i`` (0-based finite index) in fund-weight coords."""
        return self.root_system.fundamental_weight(i)

    def pairing_weight_coroot(self, lam: Sequence[int], j: int) -> int:
        """``⟨λ, α_j∨⟩`` for finite simple coroot index ``j`` (0-based)."""
        return self.root_system.pairing_weight_coroot(lam, j)

    def act_on_weight(
        self,
        w: "AffineWeylElement | FiniteWeylElement",
        lam: Sequence[int],
    ) -> Vector:
        """Act with a Weyl element on a weight in fund-weight coordinates.

        Accepts a :class:`FiniteWeylElement` or an :class:`AffineWeylElement`
        (affine elements act through ``π``, so translations act trivially).
        Delegates to :meth:`WeightLattice.act` / 
        :meth:`FiniteWeylElement.act_on_weight`.
        """
        return self.weight_lattice.act(w, lam)

    # --- quotient map π: W̃ → W ≅ W̃ / Q∨ ---------------------------------

    def project_to_finite(self, element: AffineWeylElement) -> FiniteWeylElement:
        """Project an affine element to the finite Weyl group: ``π(w, λ) = w``.

        This is the quotient homomorphism ``W̃ → W̃ / Q∨ ≅ W``.  The kernel
        is the translation subgroup ``{t_λ : λ ∈ Q∨}``.
        """
        if not isinstance(element, AffineWeylElement):
            raise TypeError("expected AffineWeylElement")
        if element.group.label != self.label:
            raise ValueError(
                f"element belongs to {element.group.label}, not {self.label}"
            )
        return element.to_finite()

    def finite_projection(self, element: AffineWeylElement) -> FiniteWeylElement:
        """Alias for :meth:`project_to_finite`."""
        return self.project_to_finite(element)

    def finite_identity(self) -> FiniteWeylElement:
        """Identity element of the finite Weyl group ``W``."""
        return FiniteWeylElement.identity(self)

    def finite_simple(self, i: int) -> FiniteWeylElement:
        """Finite simple reflection ``s_i`` with ``i`` in ``0 .. n-1``."""
        return FiniteWeylElement.simple(self, i)

    def __repr__(self) -> str:
        return f"AffineWeylGroup({self.label!r})"


__all__ = ["AffineWeylGroup"]
