"""Tests for affine (primary) and finite Schubert localization in Sym(P)."""

import pytest

from affineweyl import (
    AffineWeylGroup,
    BilleySummand,
    WeightPolynomial,
    billey_summands,
    finite_schubert_localize,
    schubert_localize,
)
from affineweyl.schubert import (
    affine_root_in_sym,
    bruhat_le_affine,
    bruhat_le_finite,
    finite_from_word,
    finite_length,
    finite_reduced_word,
    finite_root_in_sym,
    inversion_roots_finite,
    simple_root_in_sym,
)


@pytest.fixture(params=["A~1", "A~2", "A~3", "B~2"])
def W(request):
    return AffineWeylGroup.from_label(request.param)


def test_simple_diagonal_is_minus_simple_root_affine():
    """σ_{s_i}|_{s_i} = −α_i in Sym(P) for every affine simple generator."""
    for label in ("A~1", "A~2", "B~2", "C~2"):
        W = AffineWeylGroup.from_label(label)
        for i in range(W.affine_rank):
            si = W.simple(i)
            got = W.schubert_localize(si, si)
            expect = -simple_root_in_sym(W, i)
            assert got == expect, f"{label} s{i}: {got} != {expect}"
            # also via module API / word
            assert schubert_localize([i], [i], group=W) == expect


def test_simple_diagonal_is_minus_simple_root_finite():
    for label in ("A~1", "A~2", "A~3", "B~2"):
        W = AffineWeylGroup.from_label(label)
        for i in range(W.finite_rank):
            si = W.finite_simple(i)
            got = W.finite_schubert_localize(si, si)
            alpha = tuple(1 if j == i else 0 for j in range(W.finite_rank))
            expect = -finite_root_in_sym(W, alpha)
            assert got == expect


def test_identity_localizes_to_one(W):
    R = W.symmetric_algebra
    id_a = W.identity()
    for y in W.elements_up_to_length(3):
        assert W.schubert_localize(id_a, y) == R.one
        assert isinstance(W.schubert_localize(id_a, y), WeightPolynomial)


def test_vanishing_off_bruhat_affine():
    W = AffineWeylGroup.from_label("A~2")
    R = W.symmetric_algebra
    # s1 ≰ s0
    s0, s1 = W.simple(0), W.simple(1)
    assert not bruhat_le_affine(s1, s0)
    assert W.schubert_localize(s1, s0) == R.zero
    # longer not less than shorter
    y = W.from_word([0, 1, 0])
    x = W.from_word([0, 1, 0, 2])  # length 4 if reduced
    if x.length > y.length:
        assert W.schubert_localize(x, y) == R.zero


def test_vanishing_when_no_reduced_subword():
    W = AffineWeylGroup.from_label("A~1")
    R = W.symmetric_algebra
    # along word for s0*s1*s0, the letter 1 never appears as reduced subword for s1? 
    # actually s1 appears... use s0 at a pure s1-power-like element
    y = W.from_word([1, 0, 1])  # letters only create s1-type along...
    # s0 ≰ y? In A~1 Bruhat: elements are totally ordered by length in each "parity"?
    # Check concretely via Billey
    s0 = W.simple(0)
    # y = s1 s0 s1; reduced subwords of length 1: (1), (0), (1) -> s1, s0, s1
    # so s0 ≤ y and localization nonzero
    assert W.schubert_localize(s0, y) != R.zero
    # element of length 2 that is s0 s1: subwords length 1 are s0 and s1
    # Take x = s0 s1 (len 2) at y = s0 (len 1)
    x = W.from_word([0, 1])
    assert W.schubert_localize(x, W.simple(0)) == R.zero
    assert not bruhat_le_affine(x, W.simple(0))


def test_word_independence_A1():
    W = AffineWeylGroup.from_label("A~1")
    # A~1 has unique reduced words, but still check API with explicit word
    y = W.from_word([0, 1, 0, 1])
    x = W.from_word([0, 1])
    a = W.schubert_localize(x, y)
    b = W.schubert_localize(x, y, word=y.reduced_word())
    assert a == b
    assert a.ring is W.symmetric_algebra


def test_word_independence_A2_braid():
    W = AffineWeylGroup.from_label("A~2")
    y = W.from_word([1, 2, 1])
    assert y == W.from_word([2, 1, 2])
    words = ([1, 2, 1], [2, 1, 2])
    for x in W.elements_up_to_length(3):
        # only finite-support small ball; filter those ≤ y in length
        vals = [W.schubert_localize(x, y, word=wd) for wd in words]
        assert vals[0] == vals[1], f"x={x} vals={vals}"
    # also an affine braid involving s0
    y0 = W.from_word([0, 1, 0])
    assert y0 == W.from_word([1, 0, 1])
    for x in [W.identity(), W.simple(0), W.simple(1), y0]:
        assert W.schubert_localize(x, y0, word=[0, 1, 0]) == W.schubert_localize(
            x, y0, word=[1, 0, 1]
        )


def test_lives_in_sym_P():
    W = AffineWeylGroup.from_label("A~2")
    R = W.symmetric_algebra
    x = W.from_word([0, 1])
    y = W.from_word([0, 1, 2, 0])
    f = W.schubert_localize(x, y)
    assert isinstance(f, WeightPolynomial)
    assert f.ring.nvars == W.finite_rank
    assert f.ring.coefficient_ring == "Z"
    # all coeffs int (already by type)
    for _, c in f.terms.items():
        assert isinstance(c, int)


def test_billey_summands_structure():
    W = AffineWeylGroup.from_label("A~2")
    s1 = W.simple(1)
    y = W.from_word([1, 2, 1])
    parts = billey_summands(s1, y)
    assert parts
    assert all(isinstance(p, BilleySummand) for p in parts)
    total = sum((p.value for p in parts), W.symmetric_algebra.zero)
    assert total == W.schubert_localize(s1, y)


def test_A1_hand_checks():
    """Low-length hand checks in A~1 with signed Billey."""
    W = AffineWeylGroup.from_label("A~1")
    R = W.symmetric_algebra
    s0, s1 = W.simple(0), W.simple(1)
    a0 = simple_root_in_sym(W, 0)  # −θ = −2 X0
    a1 = simple_root_in_sym(W, 1)  # +2 X0
    assert a0 == -2 * R.variable(0)
    assert a1 == 2 * R.variable(0)
    assert W.schubert_localize(s0, s0) == -a0  # +2 X0
    assert W.schubert_localize(s1, s1) == -a1  # −2 X0

    # y = s0 s1; β0=α0, β1=s0(α1). Localization σ_{s0}|_y = −embed(β0)
    y = W.from_word([0, 1])
    assert W.schubert_localize(s0, y) == -a0
    assert W.schubert_localize(s1, y) == -affine_root_in_sym(
        W, s0.act_on_root(W.simple_root(1))
    )
    # diagonal length 2: product (−β0)(−β1) = embed(β0)embed(β1)
    # both finite parts are −α for word [0,1]
    assert W.schubert_localize(y, y) == a0 * affine_root_in_sym(
        W, s0.act_on_root(W.simple_root(1))
    )


def test_finite_identity_and_vanishing():
    W = AffineWeylGroup.from_label("A~3")
    R = W.symmetric_algebra
    id_f = W.finite_identity()
    # enumerate finite W via words of length ≤ 3 using finite gens
    fins = []
    seen = {id_f}
    layer = [id_f]
    fins.append(id_f)
    for _ in range(4):
        nxt = []
        for x in layer:
            for i in range(W.finite_rank):
                y = x * W.finite_simple(i)
                if y not in seen and finite_length(y) == finite_length(x) + 1:
                    seen.add(y)
                    nxt.append(y)
                    fins.append(y)
        layer = nxt
    for w in fins:
        assert finite_schubert_localize(id_f, w) == R.one
    # s0 ≰ s1 in A3
    assert finite_schubert_localize(W.finite_simple(0), W.finite_simple(1)) == R.zero
    assert not bruhat_le_finite(W.finite_simple(0), W.finite_simple(1))


def test_finite_diagonal_product_of_inversions():
    """σ_w|_w = ∏_{α∈N(w)} (−α) under the signed convention."""
    W = AffineWeylGroup.from_label("A~2")
    R = W.symmetric_algebra
    # longest element w0 = s0 s1 s0
    w0 = finite_from_word(W, [0, 1, 0])
    assert w0 == finite_from_word(W, [1, 0, 1])
    inv = inversion_roots_finite(w0)
    assert len(inv) == finite_length(w0) == 3
    prod = R.one
    for alpha in inv:
        prod = prod * (-finite_root_in_sym(W, alpha))
    assert W.finite_schubert_localize(w0, w0) == prod
    # word independence
    assert W.finite_schubert_localize(w0, w0, word=[0, 1, 0]) == W.finite_schubert_localize(
        w0, w0, word=[1, 0, 1]
    )


def test_finite_word_independence_B2():
    W = AffineWeylGroup.from_label("B~2")
    # m(0,1)=4 in B2: 0101 = 1010
    w = finite_from_word(W, [0, 1, 0, 1])
    assert w == finite_from_word(W, [1, 0, 1, 0])
    for v_word in [(), (0,), (1,), (0, 1), (1, 0), (0, 1, 0, 1)]:
        v = finite_from_word(W, v_word)
        a = W.finite_schubert_localize(v, w, word=[0, 1, 0, 1])
        b = W.finite_schubert_localize(v, w, word=[1, 0, 1, 0])
        assert a == b


def test_accepts_words_and_group_method():
    W = AffineWeylGroup.from_label("A~2")
    assert W.schubert_localize([0, 1], [0, 1, 0]) == schubert_localize(
        W.from_word([0, 1]), W.from_word([0, 1, 0])
    )


def test_alpha0_embedding_is_minus_theta():
    W = AffineWeylGroup.from_label("A~2")
    P = W.weight_lattice
    theta = W.root_system.highest_root
    assert affine_root_in_sym(W, W.simple_root(0)) == -finite_root_in_sym(W, theta)
