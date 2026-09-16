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
