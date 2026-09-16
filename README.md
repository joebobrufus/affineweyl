# affineweyl

Untwisted affine Weyl groups as a small, stdlib-only Python library.

## Representation

Elements of the affine Weyl group \(\widetilde{W}\) are stored in the
**semidirect product** realization

\[
\widetilde{W} \cong W \ltimes Q^\vee,
\]

as pairs \((w,\lambda)\) meaning the affine isometry \(x\mapsto w(x)+\lambda\)
(i.e. \(t_\lambda\circ w\)). Multiplication is

\[
(w,\lambda)\,(w',\lambda') = (ww',\,\lambda + w(\lambda')).
\]

Length uses the Iwahori–Matsumoto formula on the finite positive roots.
Equality is equality of normal forms \((w,\lambda)\), not mere word equality.

Supported types: \(\tilde{A}_n\) (\(n\ge 1\)), \(\tilde{B}_n\) (\(n\ge 2\)),
\(\tilde{C}_n\) (\(n\ge 2\)), \(\tilde{D}_n\) (\(n\ge 4\)), \(\tilde{E}_{6,7,8}\),
\(\tilde{F}_4\), \(\tilde{G}_2\). Labels look like `A~3`, `tilde_C2`, `E~6`.

Core algebra modules (`cartan`, `root_system`, `element`, `group`) perform **no
I/O**. Parsing/formatting and the CLI live in `io` / `cli`.

## Install

```bash
cd affineweyl
pip install -e ".[dev]"
```

## API (quick start)

```python
from affineweyl import AffineWeylGroup, parse_element, format_element

W = AffineWeylGroup.from_label("A~2")   # generators s0,s1,s2
s0, s1, s2 = W.generators()
w = s0 * s1 * s0
assert w.length == 3
assert W.is_reduced([0, 1, 0])
assert (s0 * s0).is_identity()
print(format_element(parse_element(W, "s0*s1*s2")))
# enumerate small balls
elts = W.elements_up_to_length(3)
# roots / coroots: α = w·α_i  ⇒  α∨ = w·α_i∨
alpha, alpha_vee = w.root_coroot_pair(0)
assert alpha_vee == w.associated_coroot(0)
# pure translations t_λ from coroot-lattice vectors
t = W.translation((1, 0))          # t_{α_1∨} in A~2
assert t.translation == (1, 0)
assert (t * W.translation((0, 1))).translation == (1, 1)
```

Translations ``t_λ`` are the elements ``(id, λ)`` in ``W ⋉ Q∨``.  Their affine
(co)root actions are the correct linear maps for ``t_λ`` (not identity
matrices); in particular ``s_0 = t_{θ∨} ∘ s_θ``.



## Finite projection π: W̃ → W

The quotient map ``π: W̃ → W̃ / Q∨ ≅ W`` drops the translation and keeps the
finite Weyl factor: ``π(w, λ) = w``.  Translations ``t_λ`` form the kernel,
and ``π`` is a group homomorphism.

```python
from affineweyl import AffineWeylGroup

W = AffineWeylGroup.from_label("A~2")
t = W.translation((1, 0))
assert t.to_finite().is_identity()          # π(t_λ) = 1
assert W.simple(1).to_finite() == W.finite_simple(0)  # π(s_1) = s_0^{finite}
# π(s_0) = s_θ (finite reflection in the highest root)
x, y = W.from_word([0, 1]), W.from_word([2, 0])
assert (x * y).to_finite() == x.to_finite() * y.to_finite()
# also: element.finite_part, W.project_to_finite(element), W.finite_projection(element)
```

The image is a :class:`FiniteWeylElement` (finite ``w_roots`` / ``w_coroots``
matrices) with multiply / compare / inverse.

## Weight lattice (finite)

Classical weights live in the fundamental-weight basis: an ``n``-tuple
``(λ_0, …, λ_{n-1})`` means ``λ = Σ λ_i ω_i``, with ``⟨ω_i, α_j∨⟩ = δ_{ij}``
(0-based indices, same as finite simple roots).

```python
from affineweyl import AffineWeylGroup

W = AffineWeylGroup.from_label("A~2")
P = W.weight_lattice
omega0 = W.fundamental_weight(0)          # (1, 0)
assert W.pairing_weight_coroot(omega0, 0) == 1
assert W.pairing_weight_coroot(omega0, 1) == 0
assert P.index_P_mod_Q == 3               # |P/Q| = n+1 for A_n
# simple root α_0 = 2ω_0 - ω_1 in fund-weight coords
assert P.from_simple_root_coords((1, 0)) == (2, -1)
assert P.is_in_root_lattice((2, -1))
assert not P.is_in_root_lattice((1, 0))   # ω_0 ∉ Q for A_2
# finite Weyl action on P (fund-weight coords)
s0 = W.finite_simple(0)
assert s0.act_on_weight(omega0) == P.act(s0, omega0)
# s_i(ω_i) = ω_i - α_i; translations act via π as id
assert W.translation((1, 0)).act_on_weight(omega0) == omega0
```

The action uses ``w_roots`` on simple-root coordinates of ``λ`` (over ``Q``),
then converts back with the Cartan matrix; equivalently
``s_i(λ) = λ - ⟨λ, α_i∨⟩ α_i`` in fund-weight coords.

## CLI

```bash
python -m affineweyl -t A~2 mul s0 s1 s0
python -m affineweyl -t C~2 length s0 s1 s0 s1
python -m affineweyl -t A~1 enumerate 4
python -m affineweyl types
```

## Layout

```
src/affineweyl/   # library
tests/            # pytest suite
```
