"""Symmetric algebra Sym(P) of the classical weight lattice as a polynomial ring.

Identification
--------------
``Sym(P) ≅ ℤ[X_0, …, X_{n-1}]``, where the indeterminate ``X_i`` corresponds
to the fundamental weight ``ω_i`` (0-based).  A weight
``λ = Σ λ_i ω_i`` with all ``λ_i ≥ 0`` gives the monomial
``X^λ = Π X_i^{λ_i}``; general lattice points (negative coordinates) are
*not* represented here — that would be the group algebra ``ℤ[P]``
(Laurent polynomials), which is deferred.

Coefficient ring
----------------
Coefficients are integers (``ℤ``).  The finite Weyl action on ``P`` preserves
the lattice, so the induced automorphisms of ``Sym(P)`` stay over ``ℤ``;
``ℚ`` coefficients are unnecessary for the ring and the ``W``-action.

``W``-action convention
-----------------------
The finite Weyl group acts by graded ``ℤ``-algebra automorphisms induced by
the linear action on ``P``.  On generators,

    w · X_i  =  linear form of  w · ω_i
             =  Σ_j (w · ω_i)_j  X_j

(extended multiplicatively).  Equivalently, thinking of polynomials as
functions on ``P*`` via ``⟨λ, μ⟩``,

    (w · f)(μ) = f(w^{-1} · μ),

so evaluation intertwines the actions.  This matches
:meth:`~affineweyl.finite.FiniteWeylElement.act_on_weight` on degree-1
forms: the linear polynomial of ``λ`` is sent to that of ``w · λ``.

Affine elements act through ``π: W̃ → W`` (translations act as the identity),
consistent with the weight action.

No I/O in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from .element import AffineWeylElement
    from .finite import FiniteWeylElement
    from .weight_lattice import WeightLattice

Exponents = Tuple[int, ...]
Coeff = int
TermDict = Dict[Exponents, Coeff]


def _zero_exponents(n: int) -> Exponents:
    return tuple(0 for _ in range(n))


def _validate_exponents(exps: Sequence[int], n: int, name: str = "exponents") -> Exponents:
    if len(exps) != n:
        raise ValueError(f"{name} has length {len(exps)}, expected {n}")
    out = []
    for i, e in enumerate(exps):
        if isinstance(e, bool) or not isinstance(e, int):
            raise TypeError(
                f"{name}[{i}] must be an int, got {type(e).__name__}"
            )
        if e < 0:
            raise ValueError(
                f"{name}[{i}] = {e} < 0; polynomial monomials require "
                "non-negative exponents (use ℤ[P] / Laurent for general weights)"
            )
        out.append(e)
    return tuple(out)


def _normalize_terms(terms: Mapping[Exponents, Coeff], n: int) -> TermDict:
    """Drop zero coeffs; validate exponent length and non-negativity."""
    out: TermDict = {}
    for exps, coeff in terms.items():
        if isinstance(coeff, bool) or not isinstance(coeff, int):
            raise TypeError(
                f"coefficient must be an int, got {type(coeff).__name__}"
            )
        if coeff == 0:
            continue
        e = _validate_exponents(exps, n)
        out[e] = out.get(e, 0) + coeff
        if out[e] == 0:
            del out[e]
    return out


@dataclass(frozen=True)
class WeightPolynomial:
    """Sparse polynomial in ``Sym(P) ≅ ℤ[X_0, …, X_{n-1}]``.

    Internally a mapping from exponent tuples to integer coefficients.
    Immutable; arithmetic returns new instances.
    """

    ring: "SymmetricAlgebra"
    _terms: Mapping[Exponents, Coeff]  # normalized, no zeros

    def __post_init__(self) -> None:
        # Ensure mapping is a plain dict snapshot (frozen)
        object.__setattr__(self, "_terms", dict(self._terms))

    # --- views -------------------------------------------------------------

    @property
    def terms(self) -> Mapping[Exponents, Coeff]:
        """Read-only view of ``exponents → coefficient`` (non-zero only)."""
        return self._terms

    @property
    def nvars(self) -> int:
        return self.ring.nvars

    def __bool__(self) -> bool:
        return bool(self._terms)

    def is_zero(self) -> bool:
        return not self._terms

    def is_constant(self) -> bool:
        if not self._terms:
            return True
        return len(self._terms) == 1 and _zero_exponents(self.nvars) in self._terms

    def coefficient(self, exponents: Sequence[int]) -> Coeff:
        e = _validate_exponents(exponents, self.nvars)
        return self._terms.get(e, 0)

    def degree(self) -> int:
        """Total degree (``-∞`` sentinel as ``-1`` for the zero polynomial)."""
        if not self._terms:
            return -1
        return max(sum(e) for e in self._terms)

    def is_homogeneous(self) -> bool:
        if not self._terms:
            return True
        deg = None
        for e in self._terms:
            d = sum(e)
            if deg is None:
                deg = d
            elif d != deg:
                return False
        return True

    def homogeneous_component(self, d: int) -> "WeightPolynomial":
        """Return the sum of terms of total degree ``d``."""
        if d < 0:
            return self.ring.zero
        parts = {e: c for e, c in self._terms.items() if sum(e) == d}
        return WeightPolynomial(self.ring, parts)

    def homogeneous_components(self) -> Dict[int, "WeightPolynomial"]:
        """Map degree → homogeneous component (non-empty only)."""
        buckets: Dict[int, TermDict] = {}
        for e, c in self._terms.items():
            d = sum(e)
            buckets.setdefault(d, {})[e] = c
        return {d: WeightPolynomial(self.ring, t) for d, t in sorted(buckets.items())}

    # --- arithmetic --------------------------------------------------------

    def _check_ring(self, other: "WeightPolynomial") -> None:
        if other.ring is not self.ring and (
            other.ring.nvars != self.ring.nvars
            or other.ring.weight_lattice is not self.ring.weight_lattice
        ):
            raise ValueError("polynomials belong to different symmetric algebras")

    def __add__(self, other: object) -> "WeightPolynomial":
        if isinstance(other, int) and not isinstance(other, bool):
            return self + self.ring.constant(other)
        if not isinstance(other, WeightPolynomial):
            return NotImplemented
        self._check_ring(other)
        out: TermDict = dict(self._terms)
        for e, c in other._terms.items():
            s = out.get(e, 0) + c
            if s == 0:
                out.pop(e, None)
            else:
                out[e] = s
        return WeightPolynomial(self.ring, out)

    def __radd__(self, other: object) -> "WeightPolynomial":
        if isinstance(other, int) and not isinstance(other, bool):
            return self.ring.constant(other) + self
        return NotImplemented

    def __neg__(self) -> "WeightPolynomial":
        return WeightPolynomial(self.ring, {e: -c for e, c in self._terms.items()})

    def __sub__(self, other: object) -> "WeightPolynomial":
        if isinstance(other, int) and not isinstance(other, bool):
            return self + self.ring.constant(-other)
        if not isinstance(other, WeightPolynomial):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: object) -> "WeightPolynomial":
        if isinstance(other, int) and not isinstance(other, bool):
            return self.ring.constant(other) + (-self)
        return NotImplemented

    def __mul__(self, other: object) -> "WeightPolynomial":
        if isinstance(other, int) and not isinstance(other, bool):
            return self.scale(other)
        if not isinstance(other, WeightPolynomial):
            return NotImplemented
        self._check_ring(other)
        if not self._terms or not other._terms:
            return self.ring.zero
        out: TermDict = {}
        n = self.nvars
        for e1, c1 in self._terms.items():
            for e2, c2 in other._terms.items():
                e = tuple(e1[i] + e2[i] for i in range(n))
                s = out.get(e, 0) + c1 * c2
                if s == 0:
                    out.pop(e, None)
                else:
                    out[e] = s
        return WeightPolynomial(self.ring, out)

    def __rmul__(self, other: object) -> "WeightPolynomial":
        if isinstance(other, int) and not isinstance(other, bool):
            return self.scale(other)
        return NotImplemented

    def scale(self, coeff: Coeff) -> "WeightPolynomial":
        if isinstance(coeff, bool) or not isinstance(coeff, int):
            raise TypeError(f"scalar must be an int, got {type(coeff).__name__}")
        if coeff == 0:
            return self.ring.zero
        if coeff == 1:
            return self
        return WeightPolynomial(self.ring, {e: c * coeff for e, c in self._terms.items()})

    def __pow__(self, exp: int) -> "WeightPolynomial":
        if not isinstance(exp, int) or isinstance(exp, bool) or exp < 0:
            raise ValueError("exponent must be a non-negative int")
        if exp == 0:
            return self.ring.one
        result = self.ring.one
        base = self
        e = exp
        while e:
            if e & 1:
                result = result * base
            base = base * base
            e >>= 1
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int) and not isinstance(other, bool):
            return self == self.ring.constant(other)
        if not isinstance(other, WeightPolynomial):
            return NotImplemented
        if self.ring.nvars != other.ring.nvars:
            return False
        if self.ring.weight_lattice is not other.ring.weight_lattice and (
            self.ring.weight_lattice.series != other.ring.weight_lattice.series
            or self.ring.weight_lattice.rank != other.ring.weight_lattice.rank
        ):
            return False
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(
            (
                self.ring.weight_lattice.series,
                self.ring.weight_lattice.rank,
                frozenset(self._terms.items()),
            )
        )

    def __iter__(self) -> Iterator[Tuple[Exponents, Coeff]]:
        return iter(sorted(self._terms.items()))

    def __repr__(self) -> str:
        if not self._terms:
            return "0"
        parts = []
        for e, c in sorted(self._terms.items(), key=lambda kv: (sum(kv[0]), kv[0])):
            mono = self._format_monomial(e)
            if mono == "1":
                parts.append(str(c))
            elif c == 1:
                parts.append(mono)
            elif c == -1:
                parts.append(f"-{mono}")
            else:
                parts.append(f"{c}*{mono}")
        # join with + / - carefully
        s = parts[0]
        for p in parts[1:]:
            if p.startswith("-"):
                s += f" - {p[1:]}"
            else:
                s += f" + {p}"
        return s

    def _format_monomial(self, e: Exponents) -> str:
        if all(x == 0 for x in e):
            return "1"
        bits = []
        for i, p in enumerate(e):
            if p == 0:
                continue
            if p == 1:
                bits.append(f"X{i}")
            else:
                bits.append(f"X{i}^{p}")
        return "*".join(bits)


@dataclass(frozen=True)
class SymmetricAlgebra:
    """Polynomial ring ``Sym(P) ≅ ℤ[X_0, …, X_{n-1}]`` on fundamental weights.

    Attach via ``P.symmetric_algebra`` or ``W.symmetric_algebra``.
    """

    weight_lattice: "WeightLattice"

    @property
    def nvars(self) -> int:
        return self.weight_lattice.rank

    @property
    def rank(self) -> int:
        return self.nvars

    @property
    def coefficient_ring(self) -> str:
        """Documented coefficient ring (always ``\"Z\"``)."""
        return "Z"

    @cached_property
    def zero(self) -> WeightPolynomial:
        return WeightPolynomial(self, {})

    @cached_property
    def one(self) -> WeightPolynomial:
        return WeightPolynomial(self, {_zero_exponents(self.nvars): 1})

    def constant(self, coeff: Coeff) -> WeightPolynomial:
        if isinstance(coeff, bool) or not isinstance(coeff, int):
            raise TypeError(f"coefficient must be an int, got {type(coeff).__name__}")
        if coeff == 0:
            return self.zero
        return WeightPolynomial(self, {_zero_exponents(self.nvars): coeff})

    def variable(self, i: int) -> WeightPolynomial:
        """Indeterminate ``X_i`` corresponding to fundamental weight ``ω_i``."""
        n = self.nvars
        if i < 0 or i >= n:
            raise IndexError(i)
        e = tuple(1 if j == i else 0 for j in range(n))
        return WeightPolynomial(self, {e: 1})

    def monomial(self, exponents: Sequence[int], coeff: Coeff = 1) -> WeightPolynomial:
        """Return ``coeff * Π X_i^{e_i}`` (exponents must be non-negative)."""
        if isinstance(coeff, bool) or not isinstance(coeff, int):
            raise TypeError(f"coefficient must be an int, got {type(coeff).__name__}")
        if coeff == 0:
            return self.zero
        e = _validate_exponents(exponents, self.nvars)
        return WeightPolynomial(self, {e: coeff})

    def from_terms(self, terms: Mapping[Sequence[int], Coeff]) -> WeightPolynomial:
        """Build a polynomial from a mapping ``exponents → coefficient``."""
        normalized = _normalize_terms(
            {tuple(e): c for e, c in terms.items()}, self.nvars
        )
        return WeightPolynomial(self, normalized)

    def from_weight(self, lam: Sequence[int]) -> WeightPolynomial:
        """Monomial ``X^λ`` for ``λ ∈ P`` with all coordinates ``≥ 0``.

        Raises ``ValueError`` if any coordinate is negative (those belong to
        the group algebra ``ℤ[P]``, not ``Sym(P)``).
        """
        return self.monomial(lam, 1)

    def linear_form(self, lam: Sequence[int]) -> WeightPolynomial:
        """Degree-1 polynomial ``Σ λ_i X_i`` for weight ``λ = Σ λ_i ω_i``.

        Coordinates may be negative (unlike :meth:`from_weight`).
        """
        n = self.nvars
        if len(lam) != n:
            raise ValueError(f"weight has length {len(lam)}, expected {n}")
        terms: TermDict = {}
        for i, a in enumerate(lam):
            if isinstance(a, bool) or not isinstance(a, int):
                raise TypeError(
                    f"weight coordinate {i} must be an int, got {type(a).__name__}"
                )
            if a == 0:
                continue
            e = tuple(1 if j == i else 0 for j in range(n))
            terms[e] = a
        return WeightPolynomial(self, terms)

    def act(
        self,
        w: "Union[FiniteWeylElement, AffineWeylElement]",
        f: WeightPolynomial,
    ) -> WeightPolynomial:
        """Return ``w · f`` (graded algebra automorphism induced by ``w`` on ``P``).

        See module docstring for the convention.  Affine elements act via ``π``.
        """
        if f.ring is not self and f.ring.nvars != self.nvars:
            raise ValueError("polynomial belongs to a different ring")
        from .element import AffineWeylElement
        from .finite import FiniteWeylElement

        if isinstance(w, AffineWeylElement):
            w = w.to_finite()
        if not isinstance(w, FiniteWeylElement):
            raise TypeError(
                "w must be a FiniteWeylElement or AffineWeylElement, "
                f"got {type(w).__name__}"
            )
        P = self.weight_lattice
        if w.group.series != P.series or w.group.finite_rank != P.rank:
            raise ValueError(
                f"element type {w.group.label} incompatible with "
                f"SymmetricAlgebra({P.series}_{P.rank})"
            )
        return self._act_finite(w, f)

    def _act_finite(self, w: "FiniteWeylElement", f: WeightPolynomial) -> WeightPolynomial:
        """Substitute ``X_i ↦ linear_form(w · ω_i)``."""
        n = self.nvars
        images = [
            self.linear_form(w.act_on_weight(self.weight_lattice.fundamental_weight(i)))
            for i in range(n)
        ]
        # Fast path: identity substitution
        if all(
            images[i].terms == {tuple(1 if j == i else 0 for j in range(n)): 1}
            for i in range(n)
        ):
            return WeightPolynomial(self, dict(f.terms))

        result = self.zero
        for exps, coeff in f.terms.items():
            mono = self.constant(coeff)
            for i, p in enumerate(exps):
                if p == 0:
                    continue
                mono = mono * (images[i] ** p)
            result = result + mono
        return result

    def __repr__(self) -> str:
        P = self.weight_lattice
        return f"SymmetricAlgebra(Z[X0..X{P.rank - 1}]; {P.series}_{P.rank})"


# Alias matching the suggested API name
WeightPolynomialRing = SymmetricAlgebra


__all__ = [
    "Exponents",
    "Coeff",
    "WeightPolynomial",
    "SymmetricAlgebra",
    "WeightPolynomialRing",
]
