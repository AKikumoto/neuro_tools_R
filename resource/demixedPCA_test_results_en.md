# dPCA Test Results

**Script**: `test/test_demixedPCA.R` — 15 tests, all PASS  
**Notes**: [demixedPCA_notes_en.md](demixedPCA_notes_en.md)

---

## Surrogate Data (`make_demo_data`)

Mirrors the Kobak 2016 `dPCA_demo.ipynb`:

- Size: `X[N=100, S=6, T=250]`
- Two latent factors: $z_t = t/T$ (time ramp), $z_s = s/S$ (stimulus spacing)
- Per-channel random coefficients $a_t, a_s \sim \mathcal{N}(0,1)$: $X[n,s,t] = a_t z_t[t] + a_s z_s[s] + \varepsilon$
- Noise: $\varepsilon \sim \mathcal{N}(0,\, 0.2^2)$

This data has additive stimulus and time effects with **no interaction** — ideal for dPCA validation.

---

## Test 1 — `get_marginalizations` structure

**Check**: 3 entries, names `c("s","t","st")`, correct index values

**Why**: For $K=2$, there are $2^2-1=3$ non-empty subsets, enumerated in order of increasing subset size then lexicographically.

---

## Test 2 — Marginalizations sum to centred $\tilde{X}$

**Check**: `max(|X_s + X_t + X_st − X_centred|) < 1e-10`

**Why**: The inclusion–exclusion construction guarantees exact reconstruction. Floating-point residual is $O(10^{-14})$.

---

## Test 3 — Marginalizations are mutually Frobenius-orthogonal

**Check**: all pairwise inner products $< 10^{-6}$

**Why**: Each $X^{(\phi)}$ varies only along axes in $\phi$ and is constant along axes in $\phi^c$. Inner products cancel by symmetry (ANOVA non-confounding principle). Threshold $10^{-6}$ accommodates floating-point accumulation over $N \times S \times T = 15\,000$ elements.

---

## Test 4 — `dpca_fit` encoder/decoder shapes

**Check**: `dim(P[[φ]]) == c(N, k)` and `dim(D[[φ]]) == c(N, k)` for all $\phi$

**Why**: $F_\phi = U_{1:k}$ and $D_\phi = C_\phi^\top U_{1:k}$, both $[N \times k]$. Shape checks catch transposition or axis-ordering bugs.

---

## Test 5 — `dpca_transform` output shapes

**Check**: `dim(Z[[φ]]) == c(k, S, T)`

**Why**: $Z_\phi = D_\phi^\top \tilde{X}_\text{flat}$ is $[k \times S \cdot T]$, reshaped to $[k \times S \times T]$. R arrays are column-major, so reshape order matters.

---

## Test 6 — Explained variance: non-negative and sorted descending

**Check**: `all(ev >= -1e-10)` and `all(diff(ev) <= 1e-10)`

**Why**: $R^2_{\phi,j} = \|D_{\phi,j}^\top \tilde{X}\|^2 / \|\tilde{X}\|_F^2 \geq 0$. SVD singular values are sorted descending, so explained variances are too.

---

## Test 7 — First stimulus component captures stimulus variance

**Check**: `var(rowMeans(z1)) > var(colMeans(z1)) * 0.5`

- `rowMeans(z1)`: per-stimulus average → stimulus-driven variance
- `colMeans(z1)`: per-timepoint average → time-driven variance

A stimulus component should have stimulus variance at least half as large as time variance (coefficient 0.5 allows for noise contamination).

---

## Test 8 — First time component captures time variance

**Check**: `var(colMeans(z1)) > var(rowMeans(z1)) * 0.5`

Mirror of Test 7: the time linear ramp $z_t$ should dominate `Z["t"]`.

---

## Test 9 — Stimulus + time EV exceeds interaction EV

**Check**: `sum(ev["s"]) + sum(ev["t"]) > sum(ev["st"])`

**Why**: Surrogate data is additive (zero interaction). $X^{(st)} \approx 0$, so interaction components explain little variance. This validates that dPCA correctly attributes most variance to the main effects.

---

## Test 10 — `dpca_inverse_transform` returns `[N, S, T]`

**Check**: `dim(Xrec) == c(N, S, T)`

**Why**: $\hat{X}^{(s)} = F_s Z_s$ is $[N \times S \cdot T]$, reshaped to $[N \times S \times T]$.  
This test previously caught a bug where `nrow` was set to `nrow(Pk)=N` instead of `ncol(Pk)=k`.

---

## Test 11 — Sum of reconstructions approximates $\tilde{X}$

**Check**: `relative_error < 0.5` with `n_components=6`

**Why**: With finite $k$, reconstruction is approximate. A 50% relative error threshold checks that the approximation is reasonable without demanding a complete basis. With more components the error decreases monotonically.

---

## Test 12 — Three-label case yields 7 marginalizations

**Check**: `length(m) == 7` and correct names for `"stc"`

**Why**: $2^3 - 1 = 7$. Names ordered by subset size then lexicographically: `"s","t","c","st","sc","tc","stc"`.

---

## Test 13 — Regularization does not change encoder/decoder shape

**Check**: `dim(P[[φ]]) == c(N, k)` with `regularizer=0.01`

**Why**: Regularization appends $\lambda I_N$ columns to $\tilde{X}_\text{flat}$ before the SVD. The SVD rank-$k$ truncation is unaffected.

---

## Test 14 — `n_components` as named list

**Check**: `ncol(P[["s"]]) == 2`, `ncol(P[["t"]]) == 4`, `ncol(P[["st"]]) == 1`

**Why**: `dpca_fit` accepts `n_components` as a named list to allow different numbers of components per marginalization. Useful when different factors need more or fewer dimensions.

---

## Test 15 — Stimulus component EV > interaction component EV

**Check**: `ev[["s"]][1] > ev[["st"]][1]`

**Why**: For additive surrogate data, the first stimulus component explains substantial variance while the interaction component explains near zero. This confirms dPCA separates structured factors from residuals.

---

## Summary Table

| # | Test | Key assertion | Basis |
|---|------|---------------|-------|
| 1 | Get marginalizations | $2^K-1$ subsets, correct names | Combinatorics |
| 2 | Sum = $\tilde{X}$ | exact reconstruction | Inclusion–exclusion |
| 3 | Frobenius orthogonality | pairwise inner products ≈ 0 | ANOVA non-confounding |
| 4 | P, D shape `[N,k]` | SVD output dimensions | Matrix multiply |
| 5 | Z shape `[k,S,T]` | reshape correctness | Column-major arrays |
| 6 | EV non-negative, sorted | $R^2 \geq 0$, SVD ordering | SVD properties |
| 7 | Z[s] captures stimulus variance | var\_s > var\_t × 0.5 | Main effect separation |
| 8 | Z[t] captures time variance | var\_t > var\_s × 0.5 | Main effect separation |
| 9 | s + t EV > st EV | additive structure | Zero interaction in data |
| 10 | Inverse transform shape | `[N,S,T]` | Matrix reshape |
| 11 | Reconstruction error < 50% | approximate reconstruction | Finite k approximation |
| 12 | 3-label: 7 marginalizations | $2^3-1=7$ | Combinatorics |
| 13 | Regularization: shape unchanged | column append before SVD | Implementation |
| 14 | Named-list n\_components | per-marginalization k | — |
| 15 | s EV > st EV | main effect > interaction | Additive data property |
