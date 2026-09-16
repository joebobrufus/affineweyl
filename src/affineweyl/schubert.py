"""Equivariant Schubert class localization via the (signed) Billey formula.

Primary object
--------------
Localizations of **affine** Schubert classes for the affine flag variety
associated to the untwisted affine Weyl group ``W̃``.  For ``x, y ∈ W̃``,
``schubert_localize(x, y)`` returns ``σ_x|_y ∈ Sym(P)``, where ``P`` is the
classical weight lattice and ``Sym(P) ≅ ℤ[X_i]`` is this package's symmetric
algebra on fundamental weights (the ``T``-equivariant cohomology of a point
for the finite maximal torus).

Signed Billey formula
---------------------
Fix a reduced word ``(i_1, …, i_ℓ)`` for ``y``.  Write

    β_k = s_{i_1} ⋯ s_{i_{k-1}}(α_{i_k})

for the positive affine root seen at position ``k``.  Then

    σ_x|_y  =  ∑_J  ∏_{k ∈ J} (− β̂_k),

summed over subsequences ``J ⊆ {1..ℓ}`` such that ``(i_j)_{j∈J}`` is a
reduced word for ``x``.  Here ``β̂_k`` denotes the image of the affine root
``β_k`` in ``Sym(P)`` (see below).

The overall minus on each root factor is the package normalization

    σ_{s_i}|_{s_i} = −α_i   (in Sym(P)),

for every affine simple reflection ``s_0, …, s_n``.  Ordinary (unsigned)
Billey would give ``+α_i`` on the diagonal of length one; we use the signed
form consistently for all summands.

Affine roots → Sym(P)
---------------------
Affine roots live in affine simple-root coordinates ``(c_0, …, c_n)`` with
``α_0 = δ − θ``.  The **finite classical part** of such a root is

    α_fin = (c_1 − c_0 θ_1, …, c_n − c_0 θ_n)   ∈ Q

(simple-root coordinates of the finite root system), equivalently the
coefficient of ``δ`` is dropped under the identification of ``T``-weights
with the finite Cartan.  In particular ``α_0 ↦ −θ`` and each finite simple
root ``α_i`` (``i ≥ 1``) maps to itself.  The image in ``Sym(P)`` is the
linear form of ``α_fin`` in fundamental-weight coordinates via
:meth:`~affineweyl.weight_lattice.WeightLattice.from_simple_root_coords`
and :meth:`~affineweyl.symmetric_algebra.SymmetricAlgebra.linear_form`.

This matches ``T``-equivariant localization for the **affine flag variety**
``Ĝ/B̂`` (Schubert classes indexed by all of ``W̃``), valued in
``H_T^*(pt) ≅ Sym(P)``.  Affine Grassmannian Schubert calculus (cosets
``W̃/W``) is not implemented separately.

Finite helper
-------------
:func:`finite_schubert_localize` applies the same signed Billey formula
inside the finite Weyl group ``W`` (finite simple roots only), for the
finite flag variety ``G/B``.

No I/O in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .element import AffineWeylElement
from .finite import FiniteWeylElement
from .root_system import Vector, _matmul
from .symmetric_algebra import WeightPolynomial

if TYPE_CHECKING:
    from .group import AffineWeylGroup

Word = Tuple[int, ...]
AffineOrWord = Union[AffineWeylElement, Sequence[int]]
FiniteOrWord = Union[FiniteWeylElement, Sequence[int]]


def affine_root_finite_part(
    group: "AffineWeylGroup",
    aff_root: Sequence[int],
) -> Vector:
    """Classical finite part of an affine root (drop the ``δ`` component).

    If ``aff_root = (c_0, …, c_n)`` in affine simple-root coordinates and
    ``θ`` is the highest root, returns
    ``(c_1 − c_0 θ_0, …, c_n − c_0 θ_{n-1})`` in finite simple-root
    coordinates.  Thus ``α_0 = δ − θ`` maps to ``−θ``.
    """
    n = group.finite_rank
    if len(aff_root) != group.affine_rank:
        raise ValueError(
            f"affine root has length {len(aff_root)}, expected {group.affine_rank}"
        )
    theta = group.root_system.highest_root
    c0 = int(aff_root[0])
    return tuple(int(aff_root[i + 1]) - c0 * theta[i] for i in range(n))


def affine_root_in_sym(
    group: "AffineWeylGroup",
    aff_root: Sequence[int],
) -> WeightPolynomial:
    """Embed an affine root into ``Sym(P)`` via its finite classical part."""
    P = group.weight_lattice
    R = group.symmetric_algebra
    fin = affine_root_finite_part(group, aff_root)
    return R.linear_form(P.from_simple_root_coords(fin))


def finite_root_in_sym(
    group: "AffineWeylGroup",
    root: Sequence[int],
) -> WeightPolynomial:
    """Embed a finite root (simple-root coords) as a linear form in ``Sym(P)``."""
    P = group.weight_lattice
    R = group.symmetric_algebra
    if len(root) != group.finite_rank:
        raise ValueError(
            f"finite root has length {len(root)}, expected {group.finite_rank}"
        )
    return R.linear_form(P.from_simple_root_coords(tuple(int(x) for x in root)))


def simple_root_in_sym(group: "AffineWeylGroup", i: int) -> WeightPolynomial:
    """Image of the affine simple root ``α_i`` in ``Sym(P)``."""
    return affine_root_in_sym(group, group.simple_root(i))


def _as_affine_element(
    group: "AffineWeylGroup",
    x: AffineOrWord,
) -> AffineWeylElement:
    if isinstance(x, AffineWeylElement):
        if x.group.label != group.label:
            raise ValueError(
                f"element belongs to {x.group.label}, not {group.label}"
            )
        return x
    if isinstance(x, (list, tuple)):
        return group.from_word(x)
    raise TypeError(
        f"expected AffineWeylElement or word, got {type(x).__name__}"
    )


def _as_finite_element(
    group: "AffineWeylGroup",
    v: FiniteOrWord,
) -> FiniteWeylElement:
    if isinstance(v, FiniteWeylElement):
        if v.group.label != group.label:
            raise ValueError(
                f"element belongs to {v.group.label}, not {group.label}"
            )
        return v
    if isinstance(v, (list, tuple)):
        return finite_from_word(group, v)
    raise TypeError(
        f"expected FiniteWeylElement or finite word, got {type(v).__name__}"
    )


def finite_length(w: FiniteWeylElement) -> int:
    """Coxeter length of a finite Weyl element."""
    return w.group.root_system.finite_length(w.w_roots)


def finite_from_word(
    group: "AffineWeylGroup",
    word: Sequence[int],
) -> FiniteWeylElement:
    """Product of finite simple reflections (indices ``0 .. n-1``)."""
    x = FiniteWeylElement.identity(group)
    n = group.finite_rank
    for raw in word:
        i = int(raw)
        if i < 0 or i >= n:
            raise IndexError(f"finite simple index {i} out of range 0..{n - 1}")
        x = x * FiniteWeylElement.simple(group, i)
    return x


def finite_is_reduced(group: "AffineWeylGroup", word: Sequence[int]) -> bool:
    """Return True iff ``word`` is a reduced expression in the finite Weyl group."""
    return finite_length(finite_from_word(group, word)) == len(word)


def finite_reduced_word(w: FiniteWeylElement) -> Word:
    """A reduced expression in finite simple indices ``0 .. n-1`` (greedy)."""
    if w.is_identity():
        return ()
    group = w.group
    word_rev: List[int] = []
    x = w
    seen_max = finite_length(x) + 1
    while not x.is_identity():
        ell = finite_length(x)
        if ell >= seen_max or ell < 0:
            raise RuntimeError("Length did not decrease while building finite reduced word")
        seen_max = ell
        found = False
        for i in range(group.finite_rank):
            y = x * FiniteWeylElement.simple(group, i)
            if finite_length(y) < ell:
                word_rev.append(i)
                x = y
                found = True
                break
        if not found:
            raise RuntimeError(
                f"No right descent found at length {ell} for finite element of {group.label}"
            )
    return tuple(reversed(word_rev))


def inversion_roots_finite(w: FiniteWeylElement) -> Tuple[Vector, ...]:
    """Inversion set ``N(w) = {α > 0 : w(α) < 0}`` in finite simple-root coords."""
    rs = w.group.root_system
    out: List[Vector] = []
    for alpha in rs.positive_roots:
        image = w.act_on_root(alpha)
        if rs.is_negative_root(image):
            out.append(alpha)
    return tuple(out)


def bruhat_le_affine(
    x: AffineWeylElement,
    y: AffineWeylElement,
    *,
    word: Optional[Sequence[int]] = None,
) -> bool:
    """Bruhat order ``x ≤ y`` via the reduced-subexpression property."""
    if x.group.label != y.group.label:
        raise ValueError("elements belong to different groups")
    if x.length > y.length:
        return False
    if x == y:
        return True
    if x.is_identity():
        return True
    w = tuple(word) if word is not None else y.reduced_word()
    group = y.group
    if word is not None:
        if not group.is_reduced(w) or group.from_word(w) != y:
            raise ValueError("word must be a reduced expression for y")
    ell = x.length
    for J in combinations(range(len(w)), ell):
        sub = tuple(w[k] for k in J)
        if group.is_reduced(sub) and group.from_word(sub) == x:
            return True
    return False


def bruhat_le_finite(
    v: FiniteWeylElement,
    w: FiniteWeylElement,
    *,
    word: Optional[Sequence[int]] = None,
) -> bool:
    """Finite Bruhat order ``v ≤ w`` via reduced subexpressions."""
    if v.group.label != w.group.label:
        raise ValueError("elements belong to different groups")
    if finite_length(v) > finite_length(w):
        return False
    if v == w or v.is_identity():
        return True
    wd = tuple(word) if word is not None else finite_reduced_word(w)
    group = w.group
    if word is not None:
        if not finite_is_reduced(group, wd) or finite_from_word(group, wd) != w:
            raise ValueError("word must be a reduced expression for w")
    ell = finite_length(v)
    for J in combinations(range(len(wd)), ell):
        sub = tuple(wd[k] for k in J)
        if finite_is_reduced(group, sub) and finite_from_word(group, sub) == v:
            return True
    return False


@dataclass(frozen=True)
class BilleySummand:
    """One summand in the signed Billey expansion.

    Attributes
    ----------
    positions :
        Increasing indices ``J`` into the reduced word for ``y``.
    subword :
        The letters ``(i_j)_{j∈J}``.
    roots :
        Affine (or finite) roots ``β_k`` for ``k∈J`` before embedding.
    value :
        The product ``∏_{k∈J} (− embed(β_k))`` in ``Sym(P)``.
    """

    positions: Tuple[int, ...]
    subword: Word
    roots: Tuple[Vector, ...]
    value: WeightPolynomial


def _billey_betas_affine(
    group: "AffineWeylGroup",
    word: Sequence[int],
) -> Tuple[Vector, ...]:
    betas: List[Vector] = []
    partial = group.identity()
    for i in word:
        betas.append(partial.act_on_root(group.simple_root(int(i))))
        partial = partial * group.simple(int(i))
    return tuple(betas)


def _billey_betas_finite(
    group: "AffineWeylGroup",
    word: Sequence[int],
) -> Tuple[Vector, ...]:
    """Finite β_k = s_{i1}…s_{i_{k-1}}(α_{i_k}) in finite simple-root coords."""
    n = group.finite_rank
    rs = group.root_system
    betas: List[Vector] = []
    # accumulate product matrix on roots
    from .root_system import _identity, _matmatmul

    M = _identity(n)
    for raw in word:
        i = int(raw)
        alpha = tuple(1 if j == i else 0 for j in range(n))
        betas.append(_matmul(M, alpha))
        M = _matmatmul(M, rs.simple_reflections_on_roots[i])
    return tuple(betas)


def billey_summands(
    x: AffineOrWord,
    y: AffineOrWord,
    *,
    group: Optional["AffineWeylGroup"] = None,
    word: Optional[Sequence[int]] = None,
) -> List[BilleySummand]:
    """Return the signed Billey summands for affine ``σ_x|_y``.

    Parameters
    ----------
    x, y :
        Affine Weyl elements, or words in affine simple indices ``0..n``.
    group :
        Required when both ``x`` and ``y`` are raw words.
    word :
        Optional reduced word for ``y``; default ``y.reduced_word()``.
    """
    if group is None:
        if isinstance(y, AffineWeylElement):
            group = y.group
        elif isinstance(x, AffineWeylElement):
            group = x.group
        else:
            raise ValueError("group is required when x and y are words")
    xx = _as_affine_element(group, x)
    yy = _as_affine_element(group, y)
    wd = tuple(word) if word is not None else yy.reduced_word()
    if not group.is_reduced(wd):
        raise ValueError("word must be reduced")
    if group.from_word(wd) != yy:
        raise ValueError("word must represent y")

    R = group.symmetric_algebra
    betas = _billey_betas_affine(group, wd)
    ell = xx.length
    out: List[BilleySummand] = []

    if xx.is_identity():
        out.append(
            BilleySummand(positions=(), subword=(), roots=(), value=R.one)
        )
        return out

    for J in combinations(range(len(wd)), ell):
        sub = tuple(wd[k] for k in J)
        if not group.is_reduced(sub):
            continue
        if group.from_word(sub) != xx:
            continue
        roots = tuple(betas[k] for k in J)
        prod = R.one
        for beta in roots:
            prod = prod * (-affine_root_in_sym(group, beta))
        out.append(
            BilleySummand(
                positions=tuple(J),
                subword=sub,
                roots=roots,
                value=prod,
            )
        )
    return out


def schubert_localize(
    x: AffineOrWord,
    y: AffineOrWord,
    *,
    group: Optional["AffineWeylGroup"] = None,
    word: Optional[Sequence[int]] = None,
) -> WeightPolynomial:
    """Affine Schubert localization ``σ_x|_y ∈ Sym(P)`` (signed Billey).

    See module docstring for the formula, sign convention
    ``σ_{s_i}|_{s_i} = −α_i``, and the affine-root embedding into ``Sym(P)``.

    Accepts :class:`AffineWeylElement` instances or words (sequences of
    affine simple indices).  Pass ``group=`` when both arguments are words.
    """
    summands = billey_summands(x, y, group=group, word=word)
    if not summands:
        if group is None:
            group = y.group if isinstance(y, AffineWeylElement) else x.group  # type: ignore[union-attr]
        return group.symmetric_algebra.zero
    total = summands[0].value.ring.zero
    for s in summands:
        total = total + s.value
    return total


# Aliases matching the suggested API names
schubert_localization = schubert_localize


def finite_billey_summands(
    v: FiniteOrWord,
    w: FiniteOrWord,
    *,
    group: Optional["AffineWeylGroup"] = None,
    word: Optional[Sequence[int]] = None,
) -> List[BilleySummand]:
    """Signed Billey summands for **finite** Schubert localization ``σ_v|_w``."""
    if group is None:
        if isinstance(w, FiniteWeylElement):
            group = w.group
        elif isinstance(v, FiniteWeylElement):
            group = v.group
        else:
            raise ValueError("group is required when v and w are words")
    vv = _as_finite_element(group, v)
    ww = _as_finite_element(group, w)
    wd = tuple(word) if word is not None else finite_reduced_word(ww)
    if not finite_is_reduced(group, wd):
        raise ValueError("word must be reduced")
    if finite_from_word(group, wd) != ww:
        raise ValueError("word must represent w")

    R = group.symmetric_algebra
    betas = _billey_betas_finite(group, wd)
    ell = finite_length(vv)
    out: List[BilleySummand] = []

    if vv.is_identity():
        out.append(
            BilleySummand(positions=(), subword=(), roots=(), value=R.one)
        )
        return out

    for J in combinations(range(len(wd)), ell):
        sub = tuple(wd[k] for k in J)
        if not finite_is_reduced(group, sub):
            continue
        if finite_from_word(group, sub) != vv:
            continue
        roots = tuple(betas[k] for k in J)
        prod = R.one
        for beta in roots:
            prod = prod * (-finite_root_in_sym(group, beta))
        out.append(
            BilleySummand(
                positions=tuple(J),
                subword=sub,
                roots=roots,
                value=prod,
            )
        )
    return out


def finite_schubert_localize(
    v: FiniteOrWord,
    w: FiniteOrWord,
    *,
    group: Optional["AffineWeylGroup"] = None,
    word: Optional[Sequence[int]] = None,
) -> WeightPolynomial:
    """Finite Schubert localization ``σ_v|_w ∈ Sym(P)`` for ``G/B`` (signed Billey).

    Same sign convention: ``σ_{s_i}|_{s_i} = −α_i`` for finite simple
    reflections (indices ``0 .. n-1``).
    """
    summands = finite_billey_summands(v, w, group=group, word=word)
    if not summands:
        if group is None:
            group = w.group if isinstance(w, FiniteWeylElement) else v.group  # type: ignore[union-attr]
        return group.symmetric_algebra.zero
    total = summands[0].value.ring.zero
    for s in summands:
        total = total + s.value
    return total


finite_schubert_localization = finite_schubert_localize


__all__ = [
    "BilleySummand",
    "affine_root_finite_part",
    "affine_root_in_sym",
    "finite_root_in_sym",
    "simple_root_in_sym",
    "finite_length",
    "finite_from_word",
    "finite_is_reduced",
    "finite_reduced_word",
    "inversion_roots_finite",
    "bruhat_le_affine",
    "bruhat_le_finite",
    "billey_summands",
    "schubert_localize",
    "schubert_localization",
    "finite_billey_summands",
    "finite_schubert_localize",
    "finite_schubert_localization",
]
