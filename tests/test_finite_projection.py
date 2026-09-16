"""Tests for the quotient homomorphism π: W̃ → W ≅ W̃ / Q∨."""

import itertools
import random

import pytest

from affineweyl import AffineWeylGroup, FiniteWeylElement
from affineweyl.root_system import _identity, _matmatmul


def _finite_word_for_matrix(rs, target_w_roots):
    """Word in finite simple reflections realizing target_w_roots."""
    I = _identity(rs.rank)
    seen = {I: ()}
    queue = [I]
    while queue:
        M = queue.pop(0)
        word = seen[M]
        if M == target_w_roots:
            return word
        for i in range(rs.rank):
            N = _matmatmul(rs.simple_reflections_on_roots[i], M)
            if N not in seen:
                seen[N] = word + (i,)
                queue.append(N)
    raise RuntimeError("matrix not in finite Weyl group")


def _s_theta_finite(W: AffineWeylGroup) -> FiniteWeylElement:
    """Finite reflection s_θ as a FiniteWeylElement (via word in s_1…s_n)."""
    s0 = W.simple(0)
    word = _finite_word_for_matrix(W.root_system, s0.w_roots)
    # Build product of finite simples matching the embedded affine word
    result = W.finite_identity()
    for i in word:
        result = result * W.finite_simple(i)
    return result


@pytest.mark.parametrize("label", ["A~1", "A~2", "A~3", "B~2", "C~2", "G~2"])
def test_pi_translation_is_identity(label):
    """π(t_λ) = 1 for several translations."""
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    vectors = [
        tuple(0 for _ in range(n)),
        tuple(1 if i == 0 else 0 for i in range(n)),
        tuple(-1 if i == n - 1 else 0 for i in range(n)),
        tuple(i - 1 for i in range(n)),
        W.root_system.highest_coroot,
    ]
    id_w = W.finite_identity()
    for lam in vectors:
        t = W.translation(lam)
        assert t.to_finite() == id_w
        assert t.finite_part.is_identity()
        assert W.project_to_finite(t).is_identity()
        assert W.finite_projection(t) == id_w


@pytest.mark.parametrize("label", ["A~2", "A~3", "B~2", "C~2", "G~2"])
def test_pi_simple_reflections(label):
    """π(s_i) for i≥1 equals finite s_{i-1}; π(s_0) equals s_θ."""
    W = AffineWeylGroup.from_label(label)
    for i in range(1, W.affine_rank):
        assert W.simple(i).to_finite() == W.finite_simple(i - 1)
        assert W.project_to_finite(W.simple(i)) == FiniteWeylElement.simple(W, i - 1)

    s0_image = W.simple(0).to_finite()
    assert s0_image == _s_theta_finite(W)
    # Not the identity (θ is a positive root)
    assert not s0_image.is_identity()
    # s_θ is an involution
    assert (s0_image * s0_image).is_identity()


@pytest.mark.parametrize("label", ["A~1", "A~2", "C~2", "G~2"])
def test_homomorphism_property(label):
    """π(xy) = π(x)π(y) for random/small products."""
    W = AffineWeylGroup.from_label(label)
    rng = random.Random(42 + hash(label) % 1000)
    gens = list(W.generators())
    # small explicit products
    for a, b in itertools.product(gens, repeat=2):
        assert (a * b).to_finite() == a.to_finite() * b.to_finite()

    # random longer words
    for _ in range(30):
        word = [rng.randrange(W.affine_rank) for _ in range(rng.randint(1, 8))]
        x = W.from_word(word[: len(word) // 2 or 1])
        y = W.from_word(word[len(word) // 2 :])
        assert W.project_to_finite(x * y) == W.project_to_finite(x) * W.project_to_finite(y)


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2"])
def test_pi_fixes_pure_finite_elements(label):
    """π(w, 0) recovers w; pure finite elements have translation=0."""
    W = AffineWeylGroup.from_label(label)
    # All products of finite simples s_1 … s_n
    for word in itertools.product(range(1, W.affine_rank), repeat=3):
        x = W.from_word(word)
        assert x.translation == tuple(0 for _ in range(W.finite_rank))
        pi_x = x.to_finite()
        # Re-embed: build affine element from finite matrices with zero translation
        # and project again — same finite matrices
        assert pi_x.w_roots == x.w_roots
        assert pi_x.w_coroots == x.w_coroots
        # finite_simple products match
        expected = W.finite_identity()
        for i in word:
            expected = expected * W.finite_simple(i - 1)
        assert pi_x == expected


@pytest.mark.parametrize("label", ["A~2", "A~3", "C~2", "G~2"])
def test_kernel_contains_all_pure_translations(label):
    """ker π contains every pure translation t_λ."""
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    id_w = W.finite_identity()
    for coords in itertools.product([-2, -1, 0, 1, 2], repeat=n):
        if n > 2 and any(abs(c) == 2 for c in coords):
            # keep product size manageable for A~3
            if sum(1 for c in coords if c != 0) > 2:
                continue
        t = W.translation(coords)
        assert W.project_to_finite(t) == id_w
        assert t.finite_part.is_identity()


def test_api_aliases_agree():
    W = AffineWeylGroup.from_label("A~2")
    x = W.from_word([0, 1, 2, 0])
    assert x.to_finite() == x.finite_part
    assert W.project_to_finite(x) == W.finite_projection(x)
    assert x.to_finite() == W.project_to_finite(x)


def test_finite_element_multiply_compare():
    W = AffineWeylGroup.from_label("A~2")
    s0f, s1f = W.finite_simple(0), W.finite_simple(1)
    assert s0f * s0f == W.finite_identity()
    assert s0f * s1f != s1f * s0f
    assert (s0f * s1f * s0f) == (s1f * s0f * s1f)  # A2 braid
    assert hash(s0f) == hash(W.finite_simple(0))


def test_wrong_group_rejected():
    W1 = AffineWeylGroup.from_label("A~2")
    W2 = AffineWeylGroup.from_label("A~3")
    with pytest.raises(ValueError, match="belongs"):
        W1.project_to_finite(W2.simple(0))


def test_readme_projection_example():
    """Smoke test matching the documented API example."""
    W = AffineWeylGroup.from_label("A~2")
    t = W.translation((1, 0))
    assert t.to_finite().is_identity()
    assert W.simple(1).to_finite() == W.finite_simple(0)
    assert W.simple(0).finite_part == _s_theta_finite(W)
    x, y = W.from_word([0, 1]), W.from_word([2, 0])
    assert (x * y).to_finite() == x.to_finite() * y.to_finite()
