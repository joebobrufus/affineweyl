from affineweyl.root_system import FiniteRootSystem


def test_A2_roots():
    rs = FiniteRootSystem.create("A", 2)
    # Φ+ = {α1, α2, α1+α2}
    assert len(rs.positive_roots) == 3
    assert rs.highest_root == (1, 1)


def test_A1_roots():
    rs = FiniteRootSystem.create("A", 1)
    assert rs.positive_roots == ((1,),)
    assert rs.highest_coroot == (1,)


def test_B2_roots():
    rs = FiniteRootSystem.create("B", 2)
    # B2 has 4 positive roots
    assert len(rs.positive_roots) == 4


def test_C2_roots():
    rs = FiniteRootSystem.create("C", 2)
    assert len(rs.positive_roots) == 4


def test_G2_roots():
    rs = FiniteRootSystem.create("G", 2)
    assert len(rs.positive_roots) == 6


def test_E6_roots():
    rs = FiniteRootSystem.create("E", 6)
    assert len(rs.positive_roots) == 36


def test_pairing_simple():
    rs = FiniteRootSystem.create("A", 2)
    e0 = (1, 0)
    assert rs.pairing(e0, e0) == 2
    assert rs.pairing(e0, (0, 1)) == -1
