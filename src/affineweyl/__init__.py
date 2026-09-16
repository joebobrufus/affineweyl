"""affineweyl — untwisted affine Weyl groups via W ⋉ Q∨.

Primary representation
----------------------
Elements are pairs ``(w, λ)`` in the semidirect product of the finite Weyl
group with the coroot lattice (affine isometries ``x ↦ w(x) + λ``).  This
gives canonical normal forms, correct group equality, and an explicit length
formula.  See :mod:`affineweyl.element` for details.

Root / coroot action
--------------------
Elements act on the affine root and coroot lattices (simple-(co)root
coordinates).  If ``α = w · α_i`` then the associated coroot is
``α∨ = w · α_i∨`` — use ``w.root_coroot_pair(i)`` or ``w.associated_coroot(i)``.

Weight lattice
--------------
The classical weight lattice ``P`` is exposed via ``W.weight_lattice`` /
``FiniteRootSystem.weight_lattice``.  Weights are integer tuples in the
fundamental-weight basis (0-based).  Finite Weyl elements act via
``FiniteWeylElement.act_on_weight`` / ``WeightLattice.act`` /
``W.act_on_weight``.  See :mod:`affineweyl.weight_lattice`.

Symmetric algebra
-----------------
``Sym(P) ≅ ℤ[X_0,…,X_{n-1}]`` (variables = fundamental weights) is
``W.symmetric_algebra`` / ``P.symmetric_algebra``.  The finite Weyl group
acts by graded algebra automorphisms via
``FiniteWeylElement.act_on_polynomial`` / ``ring.act(w, f)``.
See :mod:`affineweyl.symmetric_algebra`.  The group algebra ``ℤ[P]``
(Laurent) is not implemented.

Finite projection
-----------------
``π: W̃ → W ≅ W̃ / Q∨`` drops translations: ``element.to_finite()`` /
``W.project_to_finite(element)`` return a :class:`FiniteWeylElement`.
"""

from .element import AffineWeylElement
from .finite import FiniteWeylElement
from .group import AffineWeylGroup
from .io import format_element, format_word, parse_element, parse_word
from .cartan import list_supported_examples, parse_affine_type
from .weight_lattice import WeightLattice
from .symmetric_algebra import SymmetricAlgebra, WeightPolynomial, WeightPolynomialRing
from .root_system import FiniteRootSystem

__version__ = "0.1.0"

__all__ = [
    "AffineWeylGroup",
    "AffineWeylElement",
    "FiniteWeylElement",
    "FiniteRootSystem",
    "WeightLattice",
    "SymmetricAlgebra",
    "WeightPolynomial",
    "WeightPolynomialRing",
    "parse_word",
    "parse_element",
    "format_word",
    "format_element",
    "parse_affine_type",
    "list_supported_examples",
    "__version__",
]
