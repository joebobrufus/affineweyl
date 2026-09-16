"""Action of affine Weyl elements on roots/coroots and the α ↔ α∨ association."""

import pytest

from affineweyl import AffineWeylGroup


@pytest.mark.parametrize("label", ["A~2", "A~3", "C~2", "C~3", "B~2", "G~2"])
def test_simple_reflection_negates_own_root(label):
    W = AffineWeylGroup.from_label(label)
    for i in range(W.affine_rank):
        si = W.simple(i)
        alpha = W.simple_root(i)
        image = si.act_on_root(alpha)
        assert image == tuple(-x for x in alpha)
        assert si.act_on_coroot(W.simple_coroot(i)) == tuple(-x for x in W.simple_coroot(i))


@pytest.mark.parametrize("label", ["A~2", "C~2", "C~3"])
def test_associated_coroot_is_same_w_on_simple_coroot(label):
    """If α = w·α_i then α∨ = w·α_i∨ (user-requested association)."""
    W = AffineWeylGroup.from_label(label)
    words = [
        (0,),
        (1,),
        (0, 1),
        (1, 0),
        (0, 1, 0),
        (1, 0, 1, 0),
        (0, 1, 2) if W.affine_rank > 2 else (0, 1),
    ]
    for word in words:
        if any(i >= W.affine_rank for i in word):
            continue
        w = W.from_word(word)
        for i in range(W.affine_rank):
            root = w.image_of_simple_root(i)
            coroot = w.associated_coroot(i)
            # same API via pair
            r2, c2 = w.root_coroot_pair(i)
            assert root == r2
            assert coroot == c2
            # association: coroot must equal w · α_i∨
            assert coroot == w.act_on_coroot(W.simple_coroot(i))
            assert root == w.act_on_root(W.simple_root(i))


def test_roundtrip_association_tilde_A():
    W = AffineWeylGroup.from_label("A~3")
    w = W.from_word([0, 1, 2, 1, 0, 3])
    for i in range(4):
        alpha = w.image_of_simple_root(i)
        alpha_vee = w.associated_coroot(i)
        # applying w^{-1} recovers the simple (co)root
        assert w.inv().act_on_root(alpha) == W.simple_root(i)
        assert w.inv().act_on_coroot(alpha_vee) == W.simple_coroot(i)


def test_roundtrip_association_tilde_C():
    W = AffineWeylGroup.from_label("C~3")
    w = W.from_word([0, 1, 0, 2, 1, 3])
    for i in range(W.affine_rank):
        alpha, alpha_vee = w.root_coroot_pair(i)
        assert w.inv().act_on_root(alpha) == W.simple_root(i)
        assert w.inv().act_on_coroot(alpha_vee) == W.simple_coroot(i)


def test_identity_acts_as_identity_on_roots():
    W = AffineWeylGroup.from_label("A~2")
    e = W.identity()
    for i in range(3):
        assert e.act_on_root(W.simple_root(i)) == W.simple_root(i)
        assert e.associated_coroot(i) == W.simple_coroot(i)


def test_action_respects_multiplication():
    W = AffineWeylGroup.from_label("C~2")
    a, b = W.simple(0), W.simple(1)
    x = a * b
    root = W.simple_root(0)
    assert x.act_on_root(root) == a.act_on_root(b.act_on_root(root))
    coroot = W.simple_coroot(1)
    assert x.act_on_coroot(coroot) == a.act_on_coroot(b.act_on_coroot(coroot))
