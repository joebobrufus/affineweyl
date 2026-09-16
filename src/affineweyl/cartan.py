"""Cartan matrices, Coxeter matrices, and affine type labels.

Finite irreducible Cartan matrices and their untwisted affine extensions.
No I/O in this module.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Matrix = Tuple[Tuple[int, ...], ...]


def _mat(rows: Sequence[Sequence[int]]) -> Matrix:
    return tuple(tuple(int(x) for x in row) for row in rows)


def cartan_A(n: int) -> Matrix:
    """Finite Cartan matrix of type A_n (n >= 1)."""
    if n < 1:
        raise ValueError("A_n requires n >= 1")
    a = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
    for i in range(n - 1):
        a[i][i + 1] = -1
        a[i + 1][i] = -1
    return _mat(a)


def cartan_B(n: int) -> Matrix:
    """Finite Cartan matrix of type B_n (n >= 2). Short root alpha_n at the end.

    Bourbaki: a_{n-1,n}=-1, a_{n,n-1}=-2.
    """
    if n < 2:
        raise ValueError("B_n requires n >= 2")
    a = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
    for i in range(n - 2):
        a[i][i + 1] = -1
        a[i + 1][i] = -1
    a[n - 2][n - 1] = -1
    a[n - 1][n - 2] = -2
    return _mat(a)


def cartan_C(n: int) -> Matrix:
    """Finite Cartan matrix of type C_n (n >= 2). Long root alpha_n at the end.

    Bourbaki: a_{n-1,n}=-2, a_{n,n-1}=-1.
    """
    if n < 2:
        raise ValueError("C_n requires n >= 2")
    a = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
    for i in range(n - 2):
        a[i][i + 1] = -1
        a[i + 1][i] = -1
    a[n - 2][n - 1] = -2
    a[n - 1][n - 2] = -1
    return _mat(a)


def cartan_D(n: int) -> Matrix:
    """Finite Cartan matrix of type D_n (n >= 4)."""
    if n < 4:
        raise ValueError("D_n requires n >= 4")
    a = [[2 if i == j else 0 for j in range(n)] for i in range(n)]
    for i in range(n - 2):
        a[i][i + 1] = -1
        a[i + 1][i] = -1
    # Branch: alpha_{n-1} and alpha_n both attach to alpha_{n-2}
    a[n - 3][n - 1] = -1
    a[n - 1][n - 3] = -1
    # Disconnect the linear a[n-2][n-1] that the loop would have set incorrectly:
    # Our loop only goes to n-2 links among 0..n-2; indices n-2 and n-1 are NOT linked by loop.
    # Loop: for i in range(n-2): links i -- i+1, so links through alpha_{n-1} (0-based n-2).
    # alpha_n (0-based n-1) only linked via branch. Good.
    return _mat(a)


def cartan_E6() -> Matrix:
    # Nodes 1-2-3-4-5 with 3 also to 6 (Bourbaki numbering 1..6 -> indices 0..5)
    # Chain 0-1-2-3-4, and 2-5
    a = [[2 if i == j else 0 for j in range(6)] for i in range(6)]
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]:
        a[i][j] = -1
        a[j][i] = -1
    return _mat(a)


def cartan_E7() -> Matrix:
    # Chain 0-1-2-3-4-5, and 2-6
    a = [[2 if i == j else 0 for j in range(7)] for i in range(7)]
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6)]:
        a[i][j] = -1
        a[j][i] = -1
    return _mat(a)


def cartan_E8() -> Matrix:
    # Chain 0-1-2-3-4-5-6, and 2-7
    a = [[2 if i == j else 0 for j in range(8)] for i in range(8)]
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (2, 7)]:
        a[i][j] = -1
        a[j][i] = -1
    return _mat(a)


def cartan_F4() -> Matrix:
    # 0=1=2-3 with double bond between 1 and 2: a12=-2, a21=-1 (1 long side? )
    # Standard: alpha1-alpha2 => alpha3-alpha4 with a_{2,3}=-2, a_{3,2}=-1
    # (short roots alpha3, alpha4)
    a = [[2 if i == j else 0 for j in range(4)] for i in range(4)]
    a[0][1] = a[1][0] = -1
    a[1][2] = -1
    a[2][1] = -2
    a[2][3] = a[3][2] = -1
    return _mat(a)


def cartan_G2() -> Matrix:
    # a01=-1, a10=-3 (alpha2 short)
    a = [[2, -1], [-3, 2]]
    return _mat(a)


def transpose(m: Matrix) -> Matrix:
    n = len(m)
    return _mat([[m[j][i] for j in range(n)] for i in range(n)])


def coxeter_from_cartan(cartan: Matrix) -> Matrix:
    """Coxeter matrix m_ij from a Cartan matrix (finite or affine)."""
    n = len(cartan)
    m = [[1 if i == j else 2 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            prod = cartan[i][j] * cartan[j][i]
            if prod == 0:
                mij = 2
            elif prod == 1:
                mij = 3
            elif prod == 2:
                mij = 4
            elif prod == 3:
                mij = 6
            else:
                mij = 0  # infinity (should not occur for finite/affine crystallographic)
            m[i][j] = m[j][i] = mij
    return _mat(m)


# Dual Coxeter labels (Kac labels) a_i^vee for untwisted affine: coefficients of theta^vee
# Indexed by finite simple coroots; used to build affine Cartan.
# For simply-laced, marks = comarks.

def _finite_cartan(series: str, n: int) -> Matrix:
    series = series.upper()
    if series == "A":
        return cartan_A(n)
    if series == "B":
        return cartan_B(n)
    if series == "C":
        return cartan_C(n)
    if series == "D":
        return cartan_D(n)
    if series == "E":
        if n == 6:
            return cartan_E6()
        if n == 7:
            return cartan_E7()
        if n == 8:
            return cartan_E8()
        raise ValueError("E_n only for n in {6,7,8}")
    if series == "F":
        if n != 4:
            raise ValueError("F_n only for n=4")
        return cartan_F4()
    if series == "G":
        if n != 2:
            raise ValueError("G_n only for n=2")
        return cartan_G2()
    raise ValueError(f"Unknown series {series}")


def parse_affine_type(label: str) -> Tuple[str, int]:
    """Parse labels like 'A~3', 'tilde_A3', 'A~_3', 'tildeA_3', 'a~2' -> ('A', 3).

    The integer is the finite rank n (affine group has n+1 simple generators).
    """
    s = label.strip().replace(" ", "").replace("~", "~")
    s_lower = s.lower()
    # Normalize prefixes
    for prefix in ("tilde_", "tilde", "aff_", "affine_"):
        if s_lower.startswith(prefix):
            s = s[len(prefix) :]
            s_lower = s.lower()
            break
    # Forms: A~3, A~_3, A_3~, A3~, ~A3
    s = s.replace("_", "")
    if s.startswith("~"):
        s = s[1:] + "~"
    # Now expect like A~3 or A3~ or A3
    letters = ""
    i = 0
    while i < len(s) and s[i].isalpha():
        letters += s[i]
        i += 1
    rest = s[i:]
    rest = rest.replace("~", "")
    if not letters or not rest.isdigit():
        raise ValueError(
            f"Cannot parse affine type {label!r}; try e.g. 'A~3', 'tilde_C2', 'E~6'"
        )
    series = letters.upper()
    n = int(rest)
    return series, n


def validate_rank(series: str, n: int) -> None:
    series = series.upper()
    if series == "A" and n < 1:
        raise ValueError("Affine A~n requires n >= 1")
    if series == "B" and n < 2:
        raise ValueError("Affine B~n requires n >= 2")
    if series == "C" and n < 2:
        raise ValueError("Affine C~n requires n >= 2")
    if series == "D" and n < 4:
        raise ValueError("Affine D~n requires n >= 4")
    if series == "E" and n not in (6, 7, 8):
        raise ValueError("Affine E~n requires n in {6,7,8}")
    if series == "F" and n != 4:
        raise ValueError("Affine F~4 only")
    if series == "G" and n != 2:
        raise ValueError("Affine G~2 only")
    if series not in "ABCDEFG":
        raise ValueError(f"Unknown series {series}")


def canonical_label(series: str, n: int) -> str:
    return f"{series.upper()}~{n}"


# Affine Dynkin: which finite node(s) bond to the affine node 0, and bond type.
# For untwisted affine, node 0 attaches with the same bond as the highest root coefficients.
# Affine Cartan is built from finite Cartan + highest root / coroot data in root_system.


SUPPORTED_SERIES = ("A", "B", "C", "D", "E", "F", "G")


def list_supported_examples() -> List[str]:
    return [
        "A~1",
        "A~2",
        "A~3",
        "B~2",
        "B~3",
        "C~2",
        "C~3",
        "D~4",
        "E~6",
        "E~7",
        "E~8",
        "F~4",
        "G~2",
    ]


__all__ = [
    "Matrix",
    "cartan_A",
    "cartan_B",
    "cartan_C",
    "cartan_D",
    "cartan_E6",
    "cartan_E7",
    "cartan_E8",
    "cartan_F4",
    "cartan_G2",
    "transpose",
    "coxeter_from_cartan",
    "parse_affine_type",
    "validate_rank",
    "canonical_label",
    "_finite_cartan",
    "list_supported_examples",
    "SUPPORTED_SERIES",
]
