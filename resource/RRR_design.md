# RRR: Master Document
### Architecture + Implementation Plan (R)

> **Prerequisite: How to Use This Document**
> Before writing any code, always open and read the corresponding section of the paper.
> Every design choice includes a citation where available.
> Only move to the code after you can explain in your own words *why* this step is necessary
> and *why* this implementation choice was made.
> **Answer the comprehension check questions before advancing to the next step.**
> This document is language-specific: all implementations are in R, following the noiseTools/MCCA coding conventions.

---

## 1. Project Goals

0. **The goal is understanding**: replicate Reduced Rank Regression (RRR) step by step in R — at the level where every line maps to a specific equation in Wu & Pillow (2025)
1. Implement **standard RRR** in R: find the rank-r weight matrix W = UV^T that best predicts Y from X via least-squares
2. Implement **ridge-RRR**: replace OLS with ridge regression before the SVD step; improves performance in low-sample regimes
3. Implement **full-covariance RRR**: account for non-spherical output noise; requires iterative EM-style fitting
4. Implement **alignment metrics**: input alignment index, output alignment index, communication fraction
5. Implement **cross-validated rank selection**: choose r by held-out R² over k folds

### Research Goal

RRR asks: **which low-dimensional subspace of input-region activity most strongly predicts output-region activity?**

```
Full-rank communication:
  y_t = W^T x_t + epsilon_t   (W is m × n, unconstrained)

Low-rank (RRR) communication:
  W = UV^T   where  U ∈ R^{m × r},  V ∈ R^{n × r},  r < min(m, n)
  y_t = V(U^T x_t) + epsilon_t

U: input axes  — directions in input space that drive output
V: output axes — directions in output space that are driven
r: communication rank
```

RRR is supervised: unlike PCA or dPCA it uses the cross-region relationship, not within-region variance, to define the subspace.

### Key Mathematical Intuition

```
OLS estimate W_LS = (X^T X)^{-1} X^T Y          [eq. 6]
       captures full-rank communication (overfits when m or n is large)

RRR estimator:
  1. W_LS          — full-rank starting point
  2. PCA on XW_LS  — find directions most predictive of Y
  3. V_r           — top r eigenvectors of W_LS^T X^T X W_LS
  4. W_RRR = W_LS V_r V_r^T                       [eq. 16]

Crucially: RRR != low-rank approximation to W_LS
  RRR minimizes  ||Y - XUV^T||^2   (variance in Y)
  SVD(W_LS) minimizes  ||W_LS - UV^T||^2  (error in W)
  They agree only when X^T X ∝ I  (spherical input distribution)  [Sec 4.1]
```

---

## 2. Overarching Architecture

```
Input matrices
  X : matrix [T × m]    input-region activity (T samples, m neurons)
  Y : matrix [T × n]    output-region activity (T samples, n neurons)
  (both centered per column before fitting)

          │
          ▼
  ┌───────────────────┐
  │  rrr_fit          │   OLS/ridge → SVD → rank-r projection
  │  (X, Y, rank,     │   returns U (input axes) and V (output axes)
  │   lambda)         │
  └───────────────────┘
          │
          ▼  model: list(U, V, W, W_ls, rank, lambda)
     ┌────┴──────────────┐
     │                   │
     ▼                   ▼
rrr_transform        rrr_r2
(Y_hat = XW)         (R² on held-out data)

          │
          ▼
  ┌───────────────────┐
  │  rrr_cv_rank      │   k-fold CV over r = 1..max_rank (and lambda grid)
  │  (X, Y, ...)      │   select r* by max mean held-out R²
  └───────────────────┘

          │
          ▼
  ┌───────────────────────────────┐
  │  rrr_alignment_input          │   alpha_in  ∈ [0,1]: how aligned is U
  │  (X, W, r)                    │   with dominant input PCs?
  └───────────────────────────────┘
  ┌───────────────────────────────┐
  │  rrr_alignment_output         │   alpha_out ∈ [0,1]: how aligned is V
  │  (X, Y, W)                    │   with dominant output PCs?
  └───────────────────────────────┘
  ┌───────────────────────────────┐
  │  rrr_comm_fraction            │   CF: fraction of output variance
  │  (X, Y, W)                    │   explained by communication
  └───────────────────────────────┘
```

**Four implementation phases:**

| Phase | Goal | Key functions |
|-------|------|---------------|
| **1** | Standard RRR + ridge | `rrr_fit`, `rrr_transform`, `rrr_r2` |
| **2** | Rank + lambda selection | `rrr_cv_rank` |
| **3** | Full-covariance RRR | `rrr_fit_noniso` |
| **4** | Alignment metrics | `rrr_alignment_input`, `rrr_alignment_output`, `rrr_comm_fraction` |

---

## 3. Input Data Format

```r
# X : matrix [T × m]   rows = time samples, cols = input neurons
# Y : matrix [T × n]   rows = time samples, cols = output neurons
#
# Preprocessing (caller's responsibility, not done inside rrr_fit):
#   X <- scale(X, center = TRUE, scale = FALSE)  # center columns
#   Y <- scale(Y, center = TRUE, scale = FALSE)
#
# Time binning:
#   If input and output are recorded at the same timepoints, rows of X and Y
#   are aligned directly. If there is a known lag, shift rows of X accordingly.
#   (Wu & Pillow 2025, Sec. 7.1)
#
# Dimensions:
#   T >> max(m, n) for reliable OLS; use ridge when T is small relative to m.
```

---

## 4. Mathematics and function contracts

The derivations that stood here (standard, ridge and full-covariance RRR, and the alignment
metrics) are in [RRR_notes.md](RRR_notes.md), section by section, and the contract of every
function (inputs, outputs, algorithm, equation numbers) is the header above it in
`RRR_lib.R`. What remains from this section are its comprehension checks:

1. Why is Step 2 performed on W_LS^T X^T X W_LS rather than directly on W_LS? Hint: what does XW_LS represent geometrically, and what quantity are we trying to maximize?
2. W_RRR ≠ rank-r SVD of W_LS in general. When do they agree? Hint: examine what happens to X^T X when input activity is spherically distributed.
3. The ridge penalty on W = UV^T simplifies to a penalty only on U (not V). Why? Hint: V is constrained to be semi-orthogonal (V^T V = I).  [eq. 30]
4. Why does the fcRRR estimator reduce to standard RRR when Sigma = sigma^2 I?
5. Why can't the output alignment index be computed by the same rotation approach used for input alignment? Hint: rotating V changes Sigma_Y (the output covariance), making the bound circular.
6. The SVD of Y^T X W_LS (as in the original MATLAB code) gives the same V_r as the eigendecomposition of W_LS^T X^T X W_LS. Verify this algebraically. Hint: if W_LS^T X^T X W_LS = V D V^T, what is the SVD of Y^T X W_LS in terms of V?

---

## 5. Test Strategy

### Numerical verification against Python reference

Every function must produce output matching `original/RRR/python/fitting.py` on identical toy data.

```r
# notebook/test_RRR.R
# ---------------------------------------------------------
# 1. Generate toy data: set.seed(42); sim <- rrr_simulate(T=50, nx=5, ny=4, rank=2)
# 2. Save: write.csv(sim$X, "toy_X.csv"); write.csv(sim$Y, "toy_Y.csv")
# 3. Run Python: svd_RRR(X, Y, rnk=2, lambda_=0) → save w0, urrr, vrrr
# 4. Assert: max(abs(model$W - W_py)) < 1e-6    (up to column sign flips in U, V)
# ---------------------------------------------------------
```

### Unit tests

The tests as written and their results are in [RRR_tests.md](RRR_tests.md).

---

## 6. Implementation Order

> **Learning principle:** Core fitting first, then extensions, then metrics.
> Every step must be verified numerically against Python before proceeding.

```
─────── Phase 1: Standard RRR (start here) ───────
Step 1   R/RRR_lib.R: rrr_simulate()
         notebook/test_RRR.R: generate data, verify shape, verify Y ≈ XUV^T + E

Step 2   R/RRR_lib.R: rrr_fit()  (lambda = 0 case)
         notebook/test_RRR.R:
           - rank(W) == rank
           - V^T V ≈ I
           - W matches Python svd_RRR output (same toy data)

Step 3   R/RRR_lib.R: rrr_transform(), rrr_r2()
         notebook/test_RRR.R:
           - R² on train >= 0 and <= OLS R²
           - Y_hat ≈ XW

─────── Phase 2: Ridge + CV rank selection ───────
Step 4   R/RRR_lib.R: rrr_fit() — add ridge branch (lambda > 0)
         notebook/test_RRR.R:
           - W_ridge matches Python svd_RRR(lambda_=100) output

Step 5   R/RRR_lib.R: rrr_cv_rank()
         notebook/test_RRR.R:
           - Best rank recovers true rank from rrr_simulate (rank=2) in large-T regime
           - Ridge helps in small-T regime (reproduce Fig. 3A of Wu & Pillow)

─────── Phase 3: Full-covariance RRR ───────
Step 6   R/RRR_lib.R: rrr_fit_noniso()
         notebook/test_RRR.R:
           - W_fcRRR matches Python svd_RRR_noniso output (with known Sigma)
           - Convergence in <= 10 iterations on simulated data

─────── Phase 4: Alignment metrics ───────
Step 7   R/RRR_lib.R: rrr_alignment_input()
         notebook/test_RRR.R:
           - Extremes: alpha_in ≈ 1 and ≈ 0 for known constructions
           - Matches Python alignment_input output

Step 8   R/RRR_lib.R: rrr_alignment_output()
         notebook/test_RRR.R:
           - comm_frac ∈ [0, 1] on all test cases
           - alpha_out extremes match expectations
           - Matches Python alignment_output output
```

---

## 7. Coding Conventions

All functions follow the `noiseTools_lib.R` / demixed_jPCA style:

```r
# 1. Multiple assignment
g(U, s, Vt) %=% svd(M)
# Note: in R, svd()$v is V (not V^T); M ≈ U %*% diag(d) %*% t(V)

# 2. Function documentation format
my_func <- function(x, param = NULL) {
  # [output1, output2] = my_func(x, param) — one-line description
  #
  # output1: description and shape
  # output2: description and shape
  #
  # x:     input description and shape
  # param: optional parameter description
  # --------------------------------------------------------
}

# 3. Matrix convention: rows = samples, cols = features
#    X[T, m],  Y[T, n]  — consistent with paper notation

# 4. eigen() vs svd():
#    For PSD matrices (e.g. W^T X^T X W), use eigen()
#    eigenvalues are returned in DECREASING order in R
#    Verify: eigen(A)$values are descending

# 5. No silent fallback to pinv without logging:
#    if (rcond(XX) < 1e-10) {
#      warning("ill-conditioned XX; using pseudoinverse")
#      W_ls <- MASS::ginv(XX) %*% (t(X) %*% Y)
#    } else {
#      W_ls <- solve(XX, t(X) %*% Y)
#    }

# 6. Tests: notebook/test_RRR.R prints PASS/FAIL explicitly.
```

---

## 8. Design Principles

1. **Understanding before code.** Every function maps to specific equations in Wu & Pillow (2025). Write the equation number in a comment before the corresponding line.

2. **Test against Python reference on identical data.** Numerical agreement (max absolute error < 1e-6) required before any function is considered complete. Sign flips in U and V columns are acceptable.

3. **`rrr_fit` handles both standard and ridge RRR via `lambda`.** Lambda = 0 gives standard OLS. No separate function for ridge; this avoids divergence in logic.

4. **Full-covariance RRR is a separate function** (`rrr_fit_noniso`) because its iterative structure, non-orthogonal V, and dependency on `expm::sqrtm` represent a substantively different algorithm.

5. **Alignment metrics take W as input**, not a model object. This allows computing alignment for any weight matrix (not just from `rrr_fit`), e.g. from OLS.

6. **Cross-validation shuffles row indices**, not fold indices. Time structure is not assumed; if temporal autocorrelation is a concern, block CV should be used (caller's responsibility).

7. **Centering is the caller's responsibility.** `rrr_fit` and all metric functions receive already-centered matrices. This separates preprocessing from estimation.

---

## 9. Connection to Existing Projects

| Context | RRR role |
|---------|----------|
| dPCA/jPCA (this repo) | RRR identifies inter-region communication subspace; dPCA identifies within-region task subspaces; complementary questions |
| EEGMRI_RuleAction two-region EEG | X = region A activity (e.g. frontal), Y = region B activity (e.g. parietal); RRR finds communication subspace across rule conditions |
| EmbeddingRNN | Apply RRR to hidden layer pairs in RNN to ask whether conjunctive coding is reflected in low-dimensional inter-layer communication |

---

## 10. Open Questions

- **Matrix square root in `rrr_fit_noniso`:** `expm::sqrtm` is general but slow; for diagonal Sigma, `sqrt(diag(Sigma))` suffices. Add a diagonal-Sigma fast path?
- **Rank selection in small-T regime:** CV may favor rank = 1 spuriously when T < m. Should minimum rank = 2 be enforced for meaningful communication subspace?
- **Alignment metric symmetry:** Wu & Pillow define separate input and output alignment. Is there a joint metric that captures both simultaneously?
- **Time-lagged RRR:** For EEG data, input drives output with a lag. Optimal lag selection could be integrated into `rrr_cv_rank` as an additional hyperparameter axis.

---

## 11. References

- **Wu, B. & Pillow, J.W. (2025).** Reduced rank regression for neural communication: a tutorial for neuroscientists. *arXiv:2512.12467v1* ← primary reference for all mathematical foundations
- **Semedo, J.D. et al. (2019).** Cortical areas interact through a communication subspace. *Neuron* 102(1), 249–259. ← empirical application establishing communication subspace concept
- **Izenman, A.J. (1975).** Reduced-rank regression for the multivariate linear model. *Journal of Multivariate Analysis* 5(2), 248–264. ← original statistical paper [ref 1 in Wu & Pillow]
- **Wu & Pillow original code.** MATLAB and Python implementations in `original/RRR/` ← reference for numerical verification
- **noiseTools_lib.R** (this project). MCCA conversion from MATLAB → R. ← coding convention reference
