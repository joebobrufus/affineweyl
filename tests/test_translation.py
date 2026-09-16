"""Tests for AffineWeylGroup.translation (pure translations t_λ)."""

from collections import deque

import pytest

from affineweyl import AffineWeylGroup
from affineweyl.root_system import _identity, _matmatmul


def _finite_word_for_matrix(rs, target_w_roots):
    """Reduced-ish word in finite simple reflections realizing target_w_roots."""
    I = _identity(rs.rank)
    seen = {I: ()}
    q = deque([I])
    while q:
        M = q.popleft()
        word = seen[M]
        if M == target_w_roots:
            return word
        for i in range(rs.rank):
            N = _matmatmul(rs.simple_reflections_on_roots[i], M)
            if N not in seen:
                seen[N] = word + (i,)
                q.append(N)
    raise RuntimeError("matrix not in finite Weyl group")


def _highest_root_reflection(W: AffineWeylGroup):
    """Finite reflection s_θ embedded via s_1 … s_n."""
    s0 = W.simple(0)
    word = _finite_word_for_matrix(W.root_system, s0.w_roots)
    return W.from_word(tuple(i + 1 for i in word))


@pytest.mark.parametrize("label", ["A~1", "A~2", "A~3", "B~2", "C~2", "G~2"])
def test_translation_roundtrip_field(label):
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    for lam in (
        tuple(0 for _ in range(n)),
        tuple(1 if i == 0 else 0 for i in range(n)),
        tuple(i + 1 for i in range(n)),
        W.root_system.highest_coroot,
    ):
        t = W.translation(lam)
        assert t.translation == tuple(lam)
        assert t.w_roots == _identity(n)
        assert t.w_coroots == _identity(n)


@pytest.mark.parametrize("label", ["A~2", "C~2", "G~2", "B~3"])
def test_translation_length_formula(label):
    W = AffineWeylGroup.from_label(label)
    rs = W.root_system
    lam = rs.highest_coroot
    t = W.translation(lam)
    expected = sum(abs(rs.pairing(lam, alpha)) for alpha in rs.positive_roots)
    assert t.length == expected


@pytest.mark.parametrize("label", ["A~2", "C~3", "G~2"])
def test_multiplying_translations_adds_vectors(label):
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    lam = tuple(1 if i % 2 == 0 else -1 for i in range(n))
    mu = tuple(i - 1 for i in range(n))
    t = W.translation(lam) * W.translation(mu)
    assert t.translation == tuple(a + b for a, b in zip(lam, mu))
    assert t == W.translation(tuple(a + b for a, b in zip(lam, mu)))
    assert t.aff_roots == W.translation(t.translation).aff_roots
    assert t.aff_coroots == W.translation(t.translation).aff_coroots


@pytest.mark.parametrize("label", ["A~2", "C~2", "B~2"])
def test_s0_equals_translation_times_s_theta(label):
    """s_0 = t_{θ∨} ∘ s_θ in the representation of _make_s0."""
    W = AffineWeylGroup.from_label(label)
    theta_vee = W.root_system.highest_coroot
    t = W.translation(theta_vee)
    s_theta = _highest_root_reflection(W)
    assert W.simple(0) == t * s_theta
    # aff matrices of t are not the identity
    a = W.affine_rank
    assert t.aff_roots != _identity(a)
    assert t.aff_coroots != _identity(a)


@pytest.mark.parametrize("label", ["A~2", "C~2"])
def test_translation_action_respects_multiplication(label):
    W = AffineWeylGroup.from_label(label)
    t1 = W.translation(W.root_system.highest_coroot)
    t2 = W.translation(tuple(1 if i == 0 else 0 for i in range(W.n)))
    x = t1 * t2
    root = W.simple_root(0)
    assert x.act_on_root(root) == t1.act_on_root(t2.act_on_root(root))
    coroot = W.simple_coroot(0)
    assert x.act_on_coroot(coroot) == t1.act_on_coroot(t2.act_on_coroot(coroot))
    for i in range(W.affine_rank):
        assert x.associated_coroot(i) == x.act_on_coroot(W.simple_coroot(i))


def test_translation_invalid_length():
    W = AffineWeylGroup.from_label("A~2")
    with pytest.raises(ValueError, match="length"):
        W.translation((1,))
    with pytest.raises(ValueError, match="length"):
        W.translation((1, 0, 0))


def test_translation_non_integer_rejected():
    W = AffineWeylGroup.from_label("A~2")
    with pytest.raises(TypeError, match="int"):
        W.translation((1, 1.5))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        W.translation((1, True))  # type: ignore[arg-type]


def test_zero_translation_is_identity():
    W = AffineWeylGroup.from_label("A~3")
    t = W.translation((0, 0, 0))
    assert t.is_identity()
    assert t == W.identity()
    assert t.aff_roots == _identity(W.affine_rank)
