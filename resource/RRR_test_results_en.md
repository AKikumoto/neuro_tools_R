# RRR Test Results

**Script**: `notebook/test_RRR.R` — 16 R-only tests, all PASS  
**Notes**: [RRR_notes_en.md](RRR_notes_en.md)

---

## Shared Toy Data

All tests share a common dataset generated with `set.seed(42)`:

- `rrr_simulate(T=50, nx=5, ny=4, rank=2, sigma_noise=0.1)`
- $T = 50$ samples, $m = 5$ input neurons, $n = 4$ output neurons, true communication rank $r = 2$
- `model <- rrr_fit(X, Y, rank=2)` fitted on this data

---

## Test 1 — `rrr_simulate` returns correct shapes

**Check**: `dim(X)==c(50,5)`, `dim(Y)==c(50,4)`, `dim(U)==c(5,2)`, `dim(V)==c(4,2)`, `dim(B)==c(5,4)`

**Why**: $X \in \mathbb{R}^{T \times m}$, $Y \in \mathbb{R}^{T \times n}$, $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$, $B = UV^\top \in \mathbb{R}^{m \times n}$. Shape errors here propagate silently into wrong matrix products downstream.

---

## Test 2 — `rrr_simulate` Y ≈ XB up to noise (SNR > 5)

**Check**: `signal_var / noise_var > 5` where `signal_var = mean((X %*% B)^2)`, `noise_var = mean((Y - X %*% B)^2)`

**Why**: With `sigma_noise=0.1` and randomly generated $U, V$ of order 1, the signal variance $\text{Var}(XB) = \text{Var}(X U V^\top)$ substantially exceeds the noise variance $\sigma^2 = 0.01$. A signal-to-noise ratio > 5 ensures the simulated data is in a recoverable regime where RRR should reliably identify the communication subspace.

---

## Test 3 — `rrr_fit` W has numerical rank == r

**Check**: `sum(svd(model$W)$d > 1e-8) == 2`

**Why**: The RRR weight matrix is $W_\text{RRR} = U_\text{RRR} V_\text{RRR}^\top$ where $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{n \times r}$. A product of an $m \times r$ and an $r \times n$ matrix has rank at most $r$. This test catches bugs where the projection step `W_ls %*% V %*% t(V)` is incorrect (e.g., wrong transpose, wrong index selection).

---

## Test 4 — `rrr_fit` V is semi-orthogonal (V^T V = I)

**Check**: `max(abs(t(model$V) %*% model$V - diag(2))) < 1e-10`

**Why**: $V_r$ is constructed from the right singular vectors of $Y^\top X W_\text{LS}$, which are orthonormal by definition (columns of $V$ in the SVD $A = USV^\top$ satisfy $V^\top V = I$). Semi-orthogonality is not just a mathematical convenience — it is what makes the ridge penalty simplification $\|UV^\top\|_F^2 = \|U\|_F^2$ valid (Section 8.3 of notes). A failure here typically indicates the wrong convention: taking rows instead of columns from `svd(...)$v`.

---

## Test 5 — `rrr_fit` R² is non-negative on training data

**Check**: `rrr_r2(X, Y, model) >= 0`

**Why**: On training data, RRR always improves on the zero predictor (mean-centered $Y$), so $R^2 = 1 - \|Y - XW\|_F^2 / \|Y\|_F^2 \geq 0$ is guaranteed. (On held-out data, $R^2$ can be negative.) A negative training $R^2$ would indicate a bug in either `rrr_fit` or `rrr_r2` — most likely a transposition error causing $XW$ to have the wrong shape.

---

## Test 6 — Ridge lambda reduces training R² relative to lambda=0

**Check**: `rrr_r2(X, Y, model_ridge) <= rrr_r2(X, Y, model) + 1e-12`

**Why**: Ridge regularization shrinks the weight matrix toward zero, which reduces training fit (but improves held-out generalization). This test verifies that: (1) the ridge branch of `rrr_fit` is actually activated, and (2) the ridge estimate is being used correctly in subsequent steps. The `+ 1e-12` tolerance accounts for floating-point rounding.

---

## Test 7 — `rrr_transform` produces X %*% model$W

**Check**: `max(abs(rrr_transform(X, model) - X %*% model$W)) < machine epsilon`

**Why**: `rrr_transform` is defined as $\hat{Y} = XW$. This test is a contractual check: `rrr_transform` must be exactly equivalent to the direct matrix multiply, with no hidden centering, scaling, or projection steps. Equality is exact (not approximate) because both paths compute the same arithmetic.

---

## Test 8 — `rrr_r2` is non-negative on training data

**Check**: `rrr_r2(X, Y, model) >= 0`

**Why**: Redundant with Test 5 by design — tests `rrr_r2` directly rather than through `rrr_fit`. Catches bugs introduced if `rrr_r2` is modified independently (e.g., incorrect denominator using `||Y - mean(Y)||` vs `||Y||`). Since $Y$ is centered by convention, these are equivalent, but it is worth verifying both code paths.

---

## Test 9 — `rrr_cv_rank` recovers true rank=2 in large-T regime

**Check**: `result$best_rank == 2` with `set.seed(99)`, `T=500`, `sigma_noise=0.1`, `max_rank=4`, `n_folds=5`

**Why**: With $T = 500 \gg m = 5$, the cross-validated $R^2$ curve peaks at the true communication rank $r = 2$: adding a third rank dimension fits noise on the training folds but does not generalize to the held-out fold. This test is probabilistic; `set.seed(99)` ensures repeatability. The large-$T$ regime is chosen because rank recovery in small-$T$ settings requires ridge regularization (see Wu & Pillow Fig. 3A).

---

## Test 10 — `rrr_fit_noniso` returns correct shapes

**Check**: `dim(U)==c(5,2)`, `dim(V)==c(4,2)`, `dim(W)==c(5,4)`, `dim(Sigma)==c(4,4)`

**Why**: Full-covariance RRR returns $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$ (not semi-orthogonal in general), $W \in \mathbb{R}^{m \times n}$, and $\hat{\Sigma} \in \mathbb{R}^{n \times n}$. The $\Sigma$ shape check is especially important: a common mistake is to return the residual matrix `Y - XW` instead of its covariance.

---

## Test 11 — `rrr_fit_noniso` converges within max_iter

**Check**: `m_ni$n_iter <= 20`

**Why**: The coordinate-ascent algorithm (alternate between updating $W$ given $\hat{\Sigma}$ and updating $\hat{\Sigma}$ from residuals) converges in practice within 1–10 iterations for typical neural data (Wu & Pillow, Section 5.2). The `n_iter` field records actual iterations taken. A value of exactly 20 would indicate non-convergence and should trigger investigation of the tolerance or initialization.

---

## Test 12 — `rrr_alignment_input` is in [0, 1]

**Check**: `0 <= alpha <= 1 + 1e-10`

**Why**: By construction, $\alpha_\text{in} = (\alpha_\text{in}^\text{raw} - \alpha_\text{in}^\text{min}) / (\alpha_\text{in}^\text{max} - \alpha_\text{in}^\text{min})$ is normalized to $[0,1]$: the numerator is the gap between the actual communication variance and the minimum possible, and the denominator is the total gap. Floating-point rounding allows a small excess above 1 (hence `+ 1e-10`). A value outside $[0,1]$ indicates a bug in the singular value padding or the min/max formulas [eqs. 39–40].

---

## Test 13 — `rrr_alignment_input` ≈ 1 when W aligned with top input PC

**Check**: `rrr_alignment_input(X, W_aligned) > 1 - 1e-10`

**Construction**: `W_aligned = u_top %*% matrix(1, 1, 4)` where `u_top` is the top eigenvector of `cov(X)`. This weight matrix places all communication variance in the single direction of maximum input variance.

**Why**: When the singular vector of $W$ coincides with the top eigenvector of $\Sigma_X$, the communication variance $\alpha_\text{in}^\text{raw} = \lambda_1^2 \sigma_1^2$ equals the theoretical maximum $\alpha_\text{in}^\text{max}$, so the index is exactly 1. This test verifies that the numerator and denominator of [eq. 41] both converge to the same value, confirming that the formula handles the extreme case correctly.

---

## Test 14 — `rrr_alignment_input` ≈ 0 when W aligned with bottom input PC

**Check**: `rrr_alignment_input(X, W_antialigned) < 1e-10`

**Construction**: `W_antialigned = u_bot %*% matrix(1, 1, 4)` where `u_bot` is the *bottom* (last) eigenvector of `cov(X)`.

**Why**: When the singular vector of $W$ coincides with the *smallest* eigenvector of $\Sigma_X$, the communication variance $\alpha_\text{in}^\text{raw} = \lambda_1^2 \sigma_m^2$ equals the theoretical minimum $\alpha_\text{in}^\text{min}$, so the numerator in [eq. 41] is zero and the index is 0. Together with Test 13, this pair verifies both extremes of the normalized range.

---

## Test 15 — `rrr_alignment_output` alpha_out and comm_frac are in [0, 1]

**Check**: `0 <= alpha_out <= 1 + 1e-10` and `0 <= comm_frac <= 1 + 1e-10`

**Why**: 
- $\text{CF} = \text{Tr}[W^\top \Sigma_X W] / \text{Tr}[\Sigma_Y]$ is bounded $[0,1]$ on training data because the communication variance cannot exceed the total output variance when $W$ is fitted to the same data.
- $\alpha_\text{out}$ is normalized analogously to $\alpha_\text{in}$ via the rearrangement extremes [eqs. 45–46].
- Values outside $[0,1]$ typically indicate that $\hat{\gamma}_j^2 = \boldsymbol{\mu}_j^\top \text{cov}(XW) \boldsymbol{\mu}_j$ contains negative entries due to numerical error in `cov(X %*% W)`, which propagates into incorrect `a_max`/`a_min` bounds.

---

## Test 16 — `rrr_alignment_output` ≈ 1 when communication drives top output PC

**Check**: `result$alpha_out > 1 - 1e-10`

**Construction**: `W_top = v_X %*% t(u_Y_top)` where `v_X` is the top eigenvector of `cov(X)` and `u_Y_top` is the top eigenvector of `cov(Y)`.

**Why**: This weight matrix maps all input activity through a single direction (`v_X`) and places the resulting prediction entirely onto the top output PC (`u_Y_top`). The communicated variance $\gamma_1^2 = \boldsymbol{\mu}_1^\top \text{cov}(XW) \boldsymbol{\mu}_1$ is maximized (concentrated on the first output mode), while $\gamma_j^2 = 0$ for $j > 1$. This drives $\alpha_\text{out}^\text{raw}$ to its maximum value, so the normalized index approaches 1.

---

## Summary Table

| # | Function | Key assertion | Mathematical basis |
|---|----------|---------------|--------------------|
| 1 | `rrr_simulate` | shapes match T, m, n, r | Matrix dimension definitions |
| 2 | `rrr_simulate` | SNR > 5 | Noise variance σ² = 0.01 ≪ signal |
| 3 | `rrr_fit` | rank(W) == r | W = UV^T, rank ≤ r |
| 4 | `rrr_fit` | V^T V = I | Right singular vectors of SVD |
| 5 | `rrr_fit` | R² ≥ 0 on train | RRR ≥ zero predictor on training data |
| 6 | `rrr_fit` | ridge ↓ train R² | Regularization shrinks toward zero |
| 7 | `rrr_transform` | exact equality to X %*% W | Contractual: no hidden operations |
| 8 | `rrr_r2` | R² ≥ 0 on train | Same as Test 5, via direct call |
| 9 | `rrr_cv_rank` | best_rank == 2 (large T) | CV peak at true rank in low-noise regime |
| 10 | `rrr_fit_noniso` | shapes correct | U[m,r], V[n,r], W[m,n], Σ[n,n] |
| 11 | `rrr_fit_noniso` | n_iter ≤ 20 | Coordinate ascent converges fast |
| 12 | `rrr_alignment_input` | α_in ∈ [0, 1] | Normalized by theoretical min/max |
| 13 | `rrr_alignment_input` | α_in ≈ 1, top-PC W | Extreme case: raw = max [eq. 41] |
| 14 | `rrr_alignment_input` | α_in ≈ 0, bottom-PC W | Extreme case: raw = min [eq. 41] |
| 15 | `rrr_alignment_output` | α_out, CF ∈ [0, 1] | Normalized by theoretical min/max |
| 16 | `rrr_alignment_output` | α_out ≈ 1, top-PC W | Communication onto dominant output PC |
