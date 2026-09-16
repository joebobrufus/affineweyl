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
"""

from .element import AffineWeylElement
from .group import AffineWeylGroup
from .io import format_element, format_word, parse_element, parse_word
from .cartan import list_supported_examples, parse_affine_type

__version__ = "0.1.0"

__all__ = [
    "AffineWeylGroup",
    "AffineWeylElement",
    "parse_word",
    "parse_element",
    "format_word",
    "format_element",
    "parse_affine_type",
    "list_supported_examples",
    "__version__",
]
