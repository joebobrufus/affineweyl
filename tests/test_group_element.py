import pytest

from affineweyl import AffineWeylGroup


@pytest.fixture(params=["A~1", "A~2", "A~3", "B~2", "C~2", "G~2"])
def group(request):
    return AffineWeylGroup.from_label(request.param)


def test_identity_length(group):
    assert group.identity().length == 0
    assert group.identity().is_identity()
    assert group.identity().reduced_word() == ()


def test_simple_reflections_length_one(group):
    for i in range(group.affine_rank):
        s = group.simple(i)
        assert s.length == 1, f"s{i} has length {s.length} in {group.label}"
        assert s * s == group.identity()
        assert s.inv() == s


def test_braid_order_matches_coxeter(group):
    """(s_i s_j)^{m_ij} = 1 for finite m_ij (skip m=0 = infinity)."""
    r = group.affine_rank
    for i in range(r):
        for j in range(i + 1, r):
            m = group.m(i, j)
            if m == 0:
                continue
            si, sj = group.simple(i), group.simple(j)
            prod = group.identity()
            for k in range(2 * m):
                prod = prod * (si if k % 2 == 0 else sj)
            assert prod.is_identity(), (
                f"(s{i} s{j})^{m} failed in {group.label}: length={prod.length}"
            )


def test_multiply_associative_A2():
    W = AffineWeylGroup.from_label("A~2")
    a, b, c = W.simple(0), W.simple(1), W.simple(2)
    assert (a * b) * c == a * (b * c)


def test_inverse_product():
    W = AffineWeylGroup.from_label("A~2")
    w = W.from_word([0, 1, 2, 1])
    assert (w * w.inv()).is_identity()
    assert (w.inv() * w).is_identity()


def test_length_subadditive_and_parity():
    W = AffineWeylGroup.from_label("A~2")
    w = W.from_word([0, 1, 0, 2])
    for i in range(3):
        wi = w.right_multiply_simple(i)
        assert abs(wi.length - w.length) == 1


def test_is_reduced():
    W = AffineWeylGroup.from_label("A~1")
    assert W.is_reduced([0, 1, 0, 1])
    assert not W.is_reduced([0, 0])
    assert not W.is_reduced([0, 1, 1, 0])


def test_reduced_word_recovers_element():
    W = AffineWeylGroup.from_label("A~2")
    for word in ([0, 1, 2], [0, 1, 0], [2, 1, 2, 0, 1], [1]):
        elt = W.from_word(word)
        rw = elt.reduced_word()
        assert len(rw) == elt.length
        assert W.from_word(rw) == elt
        assert W.is_reduced(rw)


def test_A1_infinite_dihedral_lengths():
    W = AffineWeylGroup.from_label("A~1")
    # alternating words are reduced
    for n in range(0, 8):
        word = tuple(i % 2 for i in range(n))
        assert W.from_word(word).length == n


def test_hashable_in_set():
    W = AffineWeylGroup.from_label("C~2")
    s0, s1 = W.simple(0), W.simple(1)
    assert len({s0, s0, s1, s0 * s1, s1 * s0}) == 4


def test_elements_up_to_length_A1():
    W = AffineWeylGroup.from_label("A~1")
    # infinite dihedral: 1 + 2 + 2 + 2 + 2 = 9 elements of length <= 4
    elts = W.elements_up_to_length(4)
    assert len(elts) == 1 + 2 * 4
    lengths = sorted(e.length for e in elts)
    assert lengths == [0] + [k for k in range(1, 5) for _ in range(2)]


def test_elements_up_to_length_A2_small():
    W = AffineWeylGroup.from_label("A~2")
    elts = W.elements_up_to_length(2)
    # len 0: 1; len 1: 3; len 2: 6 = 10
    assert len(elts) == 10
    assert all(e.length <= 2 for e in elts)
    counts = [sum(1 for e in elts if e.length == k) for k in range(3)]
    assert counts == [1, 3, 6]


def test_left_right_multiply():
    W = AffineWeylGroup.from_label("A~2")
    w = W.simple(0)
    assert w.left_multiply_simple(1) == W.simple(1) * w
    assert w.right_multiply_simple(1) == w * W.simple(1)


def test_pow():
    W = AffineWeylGroup.from_label("A~1")
    s = W.simple(0) * W.simple(1)
    assert (s ** 0).is_identity()
    assert s ** 2 == s * s
    assert (s ** -1) * s == W.identity()


def test_exceptional_s0_length():
    for lab in ("E~6", "F~4", "G~2"):
        W = AffineWeylGroup.from_label(lab)
        assert W.simple(0).length == 1
        assert W.simple(1).length == 1


def test_D4_basics():
    W = AffineWeylGroup.from_label("D~4")
    assert W.affine_rank == 5
    for i in range(5):
        assert W.simple(i).length == 1


def test_group_equality_not_word_equality():
    W = AffineWeylGroup.from_label("A~2")
    # s0 s1 s0 = s1 s0 s1 in A~2 (m01=3? In A~2, affine is a triangle all m=3)
    a = W.from_word([0, 1, 0])
    b = W.from_word([1, 0, 1])
    assert a == b
    assert a.length == 3
