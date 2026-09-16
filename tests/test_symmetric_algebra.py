"""Tests for Sym(P) ≅ ℤ[X_0,…,X_{n-1}] and the finite Weyl action."""

import pytest

from affineweyl import (
    AffineWeylGroup,
    SymmetricAlgebra,
    WeightPolynomial,
    WeightPolynomialRing,
)


@pytest.fixture(params=["A~2", "B~2", "C~2"])
def W(request):
    return AffineWeylGroup.from_label(request.param)


def test_ring_wiring_and_alias():
    W = AffineWeylGroup.from_label("A~2")
    P = W.weight_lattice
    R = P.symmetric_algebra
    assert isinstance(R, SymmetricAlgebra)
    assert R is W.symmetric_algebra
    assert WeightPolynomialRing is SymmetricAlgebra
    assert R.coefficient_ring == "Z"
    assert R.nvars == 2


def test_variables_are_fundamental_weights(W):
    R = W.symmetric_algebra
    P = W.weight_lattice
    n = R.nvars
    for i in range(n):
        Xi = R.variable(i)
        assert Xi == R.from_weight(P.fundamental_weight(i))
        assert Xi.degree() == 1
        assert Xi.coefficient(tuple(1 if j == i else 0 for j in range(n))) == 1


def test_from_weight_rejects_negative():
    R = AffineWeylGroup.from_label("A~2").symmetric_algebra
    with pytest.raises(ValueError, match="non-negative|negative"):
        R.from_weight((1, -1))
    with pytest.raises(ValueError):
        R.monomial((-1, 0))


def test_ring_axioms_distributive():
    R = AffineWeylGroup.from_label("A~2").symmetric_algebra
    X0, X1 = R.variable(0), R.variable(1)
    f = X0 + 2 * X1
    g = X0 * X1 + R.constant(3)
    h = X0**2 - X1
    # additive abelian
    assert f + g == g + f
    assert (f + g) + h == f + (g + h)
    assert f + R.zero == f
    assert f + (-f) == R.zero
    # multiplicative monoid
    assert f * g == g * f  # commutative polynomial ring
    assert (f * g) * h == f * (g * h)
    assert f * R.one == f
    # distributive
    assert f * (g + h) == f * g + f * h
    assert (g + h) * f == g * f + h * f
    # scalar
    assert 3 * f == f + f + f
    assert 0 * f == R.zero


def test_degree_and_homogeneous():
    R = AffineWeylGroup.from_label("A~2").symmetric_algebra
    X0, X1 = R.variable(0), R.variable(1)
    assert R.zero.degree() == -1
    assert R.one.degree() == 0
    assert (X0 * X1).degree() == 2
    f = X0**2 + 3 * X0 * X1 - X1**2
    assert f.is_homogeneous()
    assert f.degree() == 2
    g = f + X0
    assert not g.is_homogeneous()
    assert g.homogeneous_component(2) == f
    assert g.homogeneous_component(1) == X0
    assert set(g.homogeneous_components()) == {1, 2}


def test_from_terms_and_equality():
    R = AffineWeylGroup.from_label("B~2").symmetric_algebra
    f = R.from_terms({(1, 0): 2, (0, 1): -1, (1, 1): 0})
    assert f == 2 * R.variable(0) - R.variable(1)
    assert f == R.from_terms({(1, 0): 2, (0, 1): -1})


def test_linear_form_matches_weight_coords():
    W = AffineWeylGroup.from_label("A~2")
    R = W.symmetric_algebra
    lam = (2, -1)  # α_0 in fund coords
    f = R.linear_form(lam)
    assert f == 2 * R.variable(0) - R.variable(1)
    assert f.degree() == 1


# --- W-action ---------------------------------------------------------------


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2", "A~3"])
def test_simple_reflection_on_linear_forms_matches_weight_action(label):
    W = AffineWeylGroup.from_label(label)
    R = W.symmetric_algebra
    n = W.finite_rank
    for i in range(n):
        si = W.finite_simple(i)
        for j in range(n):
            omega = W.fundamental_weight(j)
            # linear form of ω_j is X_j; image should be linear form of s_i(ω_j)
            image = si.act_on_polynomial(R.variable(j))
            expected = R.linear_form(si.act_on_weight(omega))
            assert image == expected
            assert R.act(si, R.variable(j)) == expected
            assert W.act_on_polynomial(si, R.variable(j)) == expected


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2"])
def test_action_is_group_homomorphism(label):
    """(xy)·f = x·(y·f); identity fixes f."""
    W = AffineWeylGroup.from_label(label)
    R = W.symmetric_algebra
    X0, X1 = R.variable(0), R.variable(1)
    f = X0**2 + 3 * X0 * X1 - 2 * X1 + R.constant(5)
    e = W.finite_identity()
    assert e.act_on_polynomial(f) == f

    s0, s1 = W.finite_simple(0), W.finite_simple(1)
    words = [
        (s0, s1),
        (s1, s0),
        (s0, s1, s0),
        (s1, s0, s1),
        (s0, s0),  # = id
    ]
    for gens in words:
        prod = e
        for g in gens:
            prod = prod * g
        nested = f
        for g in reversed(gens):
            nested = g.act_on_polynomial(nested)
        assert prod.act_on_polynomial(f) == nested


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2"])
def test_action_preserves_multiplication(label):
    W = AffineWeylGroup.from_label(label)
    R = W.symmetric_algebra
    f = R.variable(0) + 2 * R.variable(1)
    g = R.variable(0) * R.variable(1) + R.constant(1)
    for w in [W.finite_simple(0), W.finite_simple(1), W.finite_simple(0) * W.finite_simple(1)]:
        assert w.act_on_polynomial(f * g) == w.act_on_polynomial(f) * w.act_on_polynomial(g)
        assert w.act_on_polynomial(f + g) == w.act_on_polynomial(f) + w.act_on_polynomial(g)


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2", "A~3"])
def test_action_preserves_degree(label):
    W = AffineWeylGroup.from_label(label)
    R = W.symmetric_algebra
    n = R.nvars
    # build a homogeneous poly of degree 2
    f = R.zero
    for i in range(n):
        for j in range(n):
            e = [0] * n
            e[i] += 1
            e[j] += 1
            f = f + R.monomial(e, i + j + 1)
    assert f.is_homogeneous() and f.degree() == 2
    w = W.finite_simple(0)
    if n > 1:
        w = w * W.finite_simple(1)
    wf = w.act_on_polynomial(f)
    assert wf.degree() == f.degree()
    assert wf.is_homogeneous()


def test_A2_explicit_s0_on_variables():
    W = AffineWeylGroup.from_label("A~2")
    R = W.symmetric_algebra
    s0, s1 = W.finite_simple(0), W.finite_simple(1)
    X0, X1 = R.variable(0), R.variable(1)
    # s0(ω0) = (-1, 1) ⇒ s0·X0 = -X0 + X1; s0(ω1)=ω1 ⇒ s0·X1 = X1
    assert s0.act_on_polynomial(X0) == -X0 + X1
    assert s0.act_on_polynomial(X1) == X1
    # s1(ω1) = (1, -1); s1(ω0)=ω0
    assert s1.act_on_polynomial(X1) == X0 - X1
    assert s1.act_on_polynomial(X0) == X0
    # braid on a quadratic
    f = X0 * X1
    assert (s0 * s1 * s0).act_on_polynomial(f) == (s1 * s0 * s1).act_on_polynomial(f)


def test_B2_C2_explicit_action():
    WB = AffineWeylGroup.from_label("B~2")
    WC = AffineWeylGroup.from_label("C~2")
    RB, RC = WB.symmetric_algebra, WC.symmetric_algebra
    # B2: s0(ω0)=(-1,2), s1(ω1)=(1,-1)
    assert WB.finite_simple(0).act_on_polynomial(RB.variable(0)) == RB.linear_form((-1, 2))
    assert WB.finite_simple(1).act_on_polynomial(RB.variable(1)) == RB.linear_form((1, -1))
    # C2: s0(ω0)=(-1,1), s1(ω1)=(2,-1)
    assert WC.finite_simple(0).act_on_polynomial(RC.variable(0)) == RC.linear_form((-1, 1))
    assert WC.finite_simple(1).act_on_polynomial(RC.variable(1)) == RC.linear_form((2, -1))


def test_affine_via_pi_and_translations():
    W = AffineWeylGroup.from_label("A~2")
    R = W.symmetric_algebra
    f = R.variable(0) ** 2 + R.variable(1)
    t = W.translation((1, 0))
    assert t.act_on_polynomial(f) == f
    x = W.from_word([0, 1, 0])
    assert x.act_on_polynomial(f) == x.to_finite().act_on_polynomial(f)


def test_repr_smoke():
    R = AffineWeylGroup.from_label("A~2").symmetric_algebra
    f = R.variable(0) ** 2 - 2 * R.variable(0) * R.variable(1) + R.constant(3)
    s = repr(f)
    assert "X0" in s
    assert isinstance(f, WeightPolynomial)


def test_invalid_variable_index():
    R = AffineWeylGroup.from_label("A~2").symmetric_algebra
    with pytest.raises(IndexError):
        R.variable(2)
    with pytest.raises(TypeError):
        R.act("nope", R.one)  # type: ignore[arg-type]
