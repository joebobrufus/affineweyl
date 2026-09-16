"""Action of finite (and affine-via-π) Weyl elements on the weight lattice P."""

import pytest

from affineweyl import AffineWeylGroup, FiniteWeylElement


def _pairing(lam, coroot_coords):
    """⟨λ, α∨⟩ for λ in fund-weight coords and α∨ in simple-coroot coords."""
    return sum(lam[i] * coroot_coords[i] for i in range(len(lam)))


@pytest.mark.parametrize("label", ["A~1", "A~2", "A~3", "B~2", "B~3", "C~2", "C~3"])
def test_simple_reflection_on_own_fundamental_weight(label):
    """s_i(ω_i) = ω_i - α_i in fund-weight coordinates."""
    W = AffineWeylGroup.from_label(label)
    P = W.weight_lattice
    n = W.finite_rank
    for i in range(n):
        omega = P.fundamental_weight(i)
        alpha_fund = P.from_simple_root_coords(
            tuple(1 if j == i else 0 for j in range(n))
        )
        expected = tuple(omega[k] - alpha_fund[k] for k in range(n))
        si = W.finite_simple(i)
        assert si.act_on_weight(omega) == expected
        assert P.act(si, omega) == expected
        assert W.act_on_weight(si, omega) == expected


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2"])
def test_simple_reflection_fixes_orthogonal_fundamentals(label):
    """s_i(ω_j) = ω_j when ⟨ω_j, α_i∨⟩ = 0, i.e. i ≠ j."""
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    for i in range(n):
        si = W.finite_simple(i)
        for j in range(n):
            if i == j:
                continue
            omega = W.fundamental_weight(j)
            # Only fixed if Cartan off-diagonal is 0? No: s_i(ω_j) = ω_j - δ_{ij} α_i
            # so for i ≠ j, ⟨ω_j, α_i∨⟩ = 0, hence s_i(ω_j) = ω_j.
            assert si.act_on_weight(omega) == omega


@pytest.mark.parametrize("label", ["A~2", "A~3", "B~2", "C~2", "C~3"])
def test_identity_acts_as_id(label):
    W = AffineWeylGroup.from_label(label)
    e = W.finite_identity()
    n = W.finite_rank
    samples = [
        W.fundamental_weight(i) for i in range(n)
    ] + [tuple(range(1, n + 1)), tuple(0 for _ in range(n))]
    for lam in samples:
        assert e.act_on_weight(lam) == tuple(lam)


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2", "A~3"])
def test_homomorphism_property(label):
    """(xy)·λ = x·(y·λ)."""
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    gens = [W.finite_simple(i) for i in range(n)]
    words = [
        (0,),
        (0, 1) if n > 1 else (0,),
        (1, 0) if n > 1 else (0,),
        (0, 1, 0) if n > 1 else (0, 0),
        (1, 0, 1, 0) if n > 1 else (0,),
    ]
    if n > 2:
        words.append((0, 1, 2, 1))
    lam = tuple(i + 1 for i in range(n))
    for word in words:
        elts = [gens[i] for i in word]
        prod = W.finite_identity()
        for g in elts:
            prod = prod * g
        # left-to-right action of product vs nested action
        nested = lam
        for g in reversed(elts):
            nested = g.act_on_weight(nested)
        assert prod.act_on_weight(lam) == nested


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2", "A~3", "B~3", "C~3"])
def test_pairing_invariance(label):
    """⟨w·λ, w·α∨⟩ = ⟨λ, α∨⟩ for simple and a few composite elements."""
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    rs = W.root_system
    samples_w = [W.finite_identity()] + [W.finite_simple(i) for i in range(n)]
    if n >= 2:
        samples_w.append(W.finite_simple(0) * W.finite_simple(1))
        samples_w.append(W.finite_simple(1) * W.finite_simple(0) * W.finite_simple(1))
    weights = [W.fundamental_weight(i) for i in range(n)] + [
        tuple(range(1, n + 1)),
        tuple((-1) ** i * (i + 1) for i in range(n)),
    ]
    coroots = [
        tuple(1 if j == i else 0 for j in range(n)) for i in range(n)
    ] + list(rs.positive_coroots[: min(5, len(rs.positive_coroots))])
    for w in samples_w:
        for lam in weights:
            wlam = w.act_on_weight(lam)
            for alpha_vee in coroots:
                left = _pairing(lam, alpha_vee)
                right = _pairing(wlam, w.act_on_coroot(alpha_vee))
                assert left == right


@pytest.mark.parametrize("label", ["A~2", "B~2", "C~2"])
def test_affine_sugar_factors_through_pi(label):
    """affine_elt.act_on_weight == affine_elt.to_finite().act_on_weight."""
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    lam = tuple(range(1, n + 1))
    words = [[1], [1, 2] if W.affine_rank > 2 else [1, 0], [0, 1, 0]]
    for word in words:
        if any(i >= W.affine_rank for i in word):
            continue
        x = W.from_word(word)
        assert x.act_on_weight(lam) == x.to_finite().act_on_weight(lam)
        assert W.act_on_weight(x, lam) == x.to_finite().act_on_weight(lam)
        assert W.weight_lattice.act(x, lam) == x.act_on_weight(lam)


@pytest.mark.parametrize("label", ["A~1", "A~2", "B~2", "C~2", "C~3"])
def test_translations_act_trivially_via_pi(label):
    W = AffineWeylGroup.from_label(label)
    n = W.finite_rank
    lam = W.fundamental_weight(0)
    vectors = [
        tuple(0 for _ in range(n)),
        tuple(1 if i == 0 else 0 for i in range(n)),
        W.root_system.highest_coroot,
        tuple(i - 1 for i in range(n)),
    ]
    for mu in vectors:
        t = W.translation(mu)
        assert t.to_finite().is_identity()
        assert t.act_on_weight(lam) == lam
        assert W.act_on_weight(t, lam) == lam


def test_A2_explicit_s0_s1_on_fundamentals():
    W = AffineWeylGroup.from_label("A~2")
    P = W.weight_lattice
    s0, s1 = W.finite_simple(0), W.finite_simple(1)
    # α_0 = (2,-1), α_1 = (-1,2) in fund coords
    assert s0.act_on_weight((1, 0)) == (-1, 1)  # ω_0 - α_0
    assert s1.act_on_weight((0, 1)) == (1, -1)  # ω_1 - α_1
    assert s0.act_on_weight((0, 1)) == (0, 1)
    assert s1.act_on_weight((1, 0)) == (1, 0)
    # braid: s0 s1 s0 = s1 s0 s1 on weights
    w = s0 * s1 * s0
    v = s1 * s0 * s1
    for lam in [(1, 0), (0, 1), (2, -1), (1, 1)]:
        assert w.act_on_weight(lam) == v.act_on_weight(lam)


def test_B2_C2_explicit():
    WB = AffineWeylGroup.from_label("B~2")
    WC = AffineWeylGroup.from_label("C~2")
    # B2: α_0 fund = (2,-2), α_1 fund = (-1,2)
    assert WB.finite_simple(0).act_on_weight((1, 0)) == (-1, 2)
    assert WB.finite_simple(1).act_on_weight((0, 1)) == (1, -1)
    # C2: α_0 fund = (2,-1), α_1 fund = (-2,2)
    assert WC.finite_simple(0).act_on_weight((1, 0)) == (-1, 1)
    assert WC.finite_simple(1).act_on_weight((0, 1)) == (2, -1)


def test_action_preserves_root_lattice():
    W = AffineWeylGroup.from_label("A~3")
    P = W.weight_lattice
    w = W.finite_simple(0) * W.finite_simple(1) * W.finite_simple(2)
    for alpha in W.root_system.positive_roots:
        fund = P.from_simple_root_coords(alpha)
        image = w.act_on_weight(fund)
        assert P.is_in_root_lattice(image)
        # agrees with root action converted to fund coords
        acted_root = w.act_on_root(alpha)
        assert image == P.from_simple_root_coords(acted_root)


def test_invalid_weight_length():
    W = AffineWeylGroup.from_label("A~2")
    with pytest.raises(ValueError):
        W.finite_simple(0).act_on_weight((1,))
    with pytest.raises(TypeError):
        W.weight_lattice.act("not-an-element", (1, 0))  # type: ignore[arg-type]
