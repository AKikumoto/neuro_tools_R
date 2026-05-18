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

## 4. Mathematical Foundation: Standard RRR

> **Reference:** Wu & Pillow (2025), Sections 2–3

### The regression model

```
y_t = W^T x_t + epsilon_t                          [eq. 1]

epsilon_t ~ N(0, sigma^2 I_n)   (isotropic noise)  [eq. 2]

X ∈ R^{T × m},  Y ∈ R^{T × n}
```

### OLS (full-rank) estimate

```
W_LS = (X^T X)^{-1} X^T Y                          [eq. 14]

This minimizes  ||Y - XW||^2_F  (sum of squared errors)
```

### RRR estimator (rank r)

```
Step 1: Compute W_LS (eq. 14)

Step 2: PCA on XW_LS — find top r eigenvectors of
        W_LS^T X^T X W_LS                           [eq. 15]
        → V_r = [v_1 ... v_r]   (n × r, orthonormal columns)

Step 3: Project W_LS onto V_r subspace
        W_RRR = W_LS V_r V_r^T                      [eq. 16]

Factored form:
        U_RRR = W_LS V_r   (m × r, input axes)      [eq. 17]
        V_RRR = V_r        (n × r, output axes, V^T V = I)
        W_RRR = U_RRR V_RRR^T
```

> **Comprehension Check:** Why is Step 2 performed on W_LS^T X^T X W_LS rather than directly on W_LS?
> Hint: what does XW_LS represent geometrically, and what quantity are we trying to maximize?

> **Comprehension Check:** W_RRR ≠ rank-r SVD of W_LS in general. When do they agree?
> Hint: examine what happens to X^T X when input activity is spherically distributed.

---

## 5. Mathematical Foundation: Ridge-RRR

> **Reference:** Wu & Pillow (2025), Section 5.1

### Motivation

When T is small relative to m, X^T X is ill-conditioned and W_LS overfits. Ridge regularization penalizes ||W||^2_F.

### Ridge-RRR estimator

```
Replace W_LS with ridge estimate:
  W_ridge = (X^T X + lambda I)^{-1} X^T Y           [see eq. 33, Appendix C.1]

Then proceed exactly as standard RRR (Steps 2–3) with W_ridge in place of W_LS:
  V_RRRR = top r eigenvectors of  W_ridge^T X^T X W_ridge
  U_RRRR = W_ridge V_RRRR
  W_RRRR = U_RRRR V_RRRR^T                          [eqs. 32–33]

lambda is selected by cross-validation on held-out R².
```

> **Implementation note:** `rrr_fit` handles both cases via the `lambda` argument (default 0).
> When `lambda = 0`, ridge-RRR reduces to standard RRR.

> **Comprehension Check:** The ridge penalty on W = UV^T simplifies to a penalty only on U (not V).
> Why? Hint: V is constrained to be semi-orthogonal (V^T V = I).  [eq. 30]

---

## 6. Mathematical Foundation: Full-Covariance RRR

> **Reference:** Wu & Pillow (2025), Section 5.2

### Motivation

Standard RRR assumes isotropic noise (sigma^2 I). When output neurons have correlated or unequal noise, this misspecification biases the estimate.

### Full-covariance RRR estimator

```
Model:  y_t | x_t ~ N(W^T x_t, Sigma)              [eq. 34]

Loss:  L = Tr[(Y - XUV^T) Sigma^{-1} (Y - XUV^T)^T]  [eq. 35]

Closed-form solution given Sigma:
  C_sqrt     = Sigma^{1/2}
  C_inv_sqrt = Sigma^{-1/2}
  V_fcRRR    = C_sqrt * eig_top_r(C_inv_sqrt W_LS^T X^T X W_LS C_inv_sqrt)  [eq. 36]
  U_fcRRR    = W_LS Sigma^{-1} V_fcRRR                                        [eq. 37]

When Sigma is unknown, use coordinate ascent:
  1. Initialize Sigma = I
  2. Estimate W_fcRRR given current Sigma
  3. Update Sigma = cov(Y - XW_fcRRR)   (residual covariance)
  4. Repeat until convergence  (typically 1-10 iterations)
```

> **Comprehension Check:** Why does the fcRRR estimator reduce to standard RRR when Sigma = sigma^2 I?

---

## 7. Mathematical Foundation: Alignment Metrics

> **Reference:** Wu & Pillow (2025), Section 6

### Input alignment index

Measures how much communication is aligned with dominant modes of input population.

```
Raw communication variance:
  alpha_in^raw = Tr[W^T Sigma_X W]                  [eq. 38]
  where Sigma_X = (1/T) X^T X

Maximal possible (communication aligned with top r input PCs):
  alpha_in^max = sum_{i=1}^{r} lambda_i^2 sigma_i^2  [eq. 39]
  where lambda_i = singular values of W (descending)
        sigma_i^2 = eigenvalues of Sigma_X (descending)

Minimal possible (aligned with bottom r input PCs):
  alpha_in^min = sum_{i=1}^{r} lambda_i^2 sigma_{m+1-i}^2  [eq. 40]

Normalized index:
  alpha_in = (alpha_in^raw - alpha_in^min) / (alpha_in^max - alpha_in^min)  [eq. 41]
  Range: [0, 1]
```

### Communication fraction

```
CF = Tr[W^T Sigma_X W] / Tr[Sigma_Y]               [eq. 42]
   = total communication variance / total output variance
   Range: [0, 1] on training data
```

### Output alignment index

Measures how much communication variance falls on dominant modes of output population.

```
PCA of output:   Sigma_Y = (1/T) Y^T Y → eigenvectors {mu_j}, eigenvalues {sigma_j^2}

Communicated variance along output mode j:
  gamma_j^2 = mu_j^T (W^T Sigma_X W) mu_j           [eq. 43]

Raw output alignment:
  alpha_out^raw = sum_j gamma_j^2 * sigma_j^2        [eq. 44]

Normalize to [0,1] using max/min over rearrangements of {gamma_j^2}
that preserve sum(gamma_j^2) = Gamma (total communication variance):
  [eqs. 45–46 give alpha_out^max and alpha_out^min]

  alpha_out = (alpha_out^raw - alpha_out^min) / (alpha_out^max - alpha_out^min)  [eq. 47]
```

> **Comprehension Check:** Why can't the output alignment index be computed by the same rotation approach used for input alignment?
> Hint: rotating V changes Sigma_Y (the output covariance), making the bound circular.

---

## 8. `R/RRR_lib.R` — Function Specifications

### `rrr_fit(X, Y, rank, lambda)`

```r
rrr_fit <- function(X, Y, rank, lambda = 0) {
  # [model] = rrr_fit(X, Y, rank, lambda)
  # Fit reduced rank regression with optional ridge regularization.
  #
  # model: list with elements
  #   $U     : input axes  [m × r]
  #   $V     : output axes [n × r], V^T V = I (semi-orthogonal)
  #   $W     : full weight matrix [m × n] = U %*% t(V)
  #   $W_ls  : ridge/OLS estimate [m × n]   (stored for alignment metrics)
  #   $rank  : r
  #   $lambda: lambda used
  #
  # X     : matrix [T × m], centered
  # Y     : matrix [T × n], centered
  # rank  : integer r, communication dimensionality
  # lambda: ridge penalty (0 = standard OLS)
  #
  # Algorithm:
  #   1. XX    <- t(X) %*% X + lambda * diag(ncol(X))    [eq. 14 / ridge]
  #   2. W_ls  <- solve(XX, t(X) %*% Y)                  [m × n]
  #   3. M     <- t(W_ls) %*% t(X) %*% X %*% W_ls       [n × n, PSD]
  #      g(vecs, vals) %=% eigen(M)
  #      V_r   <- vecs[, 1:rank]                         [n × r]
  #   4. U     <- W_ls %*% V_r                           [m × r]
  #      W     <- U %*% t(V_r)                           [m × n]
  # --------------------------------------------------------
  if (rank >= min(nrow(X), ncol(X), ncol(Y))) {
    stop("rank must be < min(T, m, n)")
  }
}
```

> **Comprehension Check:** The SVD of Y^T X W_LS (as in the original MATLAB code) gives the same V_r as the eigendecomposition of W_LS^T X^T X W_LS. Verify this algebraically.
> Hint: if W_LS^T X^T X W_LS = V D V^T, what is the SVD of Y^T X W_LS in terms of V?

### `rrr_transform(X, model)`

```r
rrr_transform <- function(X, model) {
  # [Y_hat] = rrr_transform(X, model)
  # Predict output from input using fitted RRR model.
  #
  # Y_hat: matrix [T × n]
  #
  # X    : matrix [T × m], centered (same centering as in rrr_fit)
  # model: output of rrr_fit
  #
  # Y_hat = X %*% model$W
  # --------------------------------------------------------
}
```

### `rrr_r2(X, Y, model)`

```r
rrr_r2 <- function(X, Y, model) {
  # [r2] = rrr_r2(X, Y, model)
  # Compute R² (fraction of Y variance explained by RRR prediction).
  #
  # r2: scalar ∈ [0, 1] on training data; may be negative on held-out data
  #
  # r2 = 1 - ||Y - X W||^2_F / ||Y||^2_F
  # --------------------------------------------------------
}
```

### `rrr_cv_rank(X, Y, max_rank, lambda_grid, n_folds)`

```r
rrr_cv_rank <- function(X, Y, max_rank, lambda_grid = 0, n_folds = 10) {
  # [best_rank, best_lambda, cv_r2] = rrr_cv_rank(X, Y, ...)
  # Select communication rank (and ridge lambda) by k-fold cross-validation.
  #
  # best_rank  : integer
  # best_lambda: numeric
  # cv_r2      : matrix [max_rank × length(lambda_grid)], mean held-out R²
  #
  # X          : matrix [T × m], centered
  # Y          : matrix [T × n], centered
  # max_rank   : integer, maximum rank to evaluate
  # lambda_grid: numeric vector of lambda values to try (default = 0)
  # n_folds    : integer
  #
  # Algorithm:
  #   Shuffle row indices; split into n_folds equal blocks.
  #   For each fold, train on remaining folds; evaluate rrr_r2 on held-out fold.
  #   For each (rank, lambda) pair, record mean held-out R².
  #   Return argmax over (rank, lambda).
  # --------------------------------------------------------
}
```

### `rrr_fit_noniso(X, Y, rank, max_iter, tol)`

```r
rrr_fit_noniso <- function(X, Y, rank, max_iter = 20, tol = 1e-6) {
  # [model] = rrr_fit_noniso(X, Y, rank)
  # Full-covariance RRR with iterative estimation of noise covariance Sigma.
  #
  # model: list with elements
  #   $U     : input axes  [m × r]
  #   $V     : output axes [n × r]  (NOT semi-orthogonal in general)
  #   $W     : weight matrix [m × n] = U %*% t(V)
  #   $Sigma : estimated noise covariance [n × n]
  #   $rank  : r
  #   $n_iter: number of iterations until convergence
  #
  # Algorithm (coordinate ascent on Sigma):
  #   1. Initialize Sigma <- diag(ncol(Y))
  #   2. Iterate:
  #      a. Compute W_ls = solve(t(X)%*%X, t(X)%*%Y)
  #      b. C_sqrt     <- expm::sqrtm(Sigma)
  #         C_inv_sqrt <- solve(C_sqrt)
  #      c. SVD of  C_inv_sqrt %*% t(Y)%*%X%*%W_ls%*%C_inv_sqrt  → V_r (top r)
  #         V <- C_sqrt %*% V_r                                    [eq. 36]
  #         U <- solve(t(X)%*%X, t(X)%*%Y%*%solve(Sigma)%*%V)   [eq. 37]
  #         W <- U %*% t(V)
  #      d. Update Sigma <- cov(Y - X %*% W)
  #      e. Check convergence: max(abs(W_new - W_old)) < tol
  # --------------------------------------------------------
  if (rank >= min(nrow(X), ncol(X), ncol(Y))) {
    stop("rank must be < min(T, m, n)")
  }
}
```

### `rrr_alignment_input(X, W, r)`

```r
rrr_alignment_input <- function(X, W, r = NULL, C = NULL) {
  # [alpha_in] = rrr_alignment_input(X, W, r, C)
  # Compute input alignment index: how much communication aligns with dominant input PCs.
  #
  # alpha_in: scalar ∈ [0, 1]   (1 = maximally aligned with top PCs)
  #
  # X: matrix [T × m], centered input activity
  # W: weight matrix [m × n]  (e.g. model$W from rrr_fit)
  # r: rank used (optional; if NULL, use rank(W))
  # C: input covariance [m × m] (optional; if NULL, compute cov(X))
  #
  # Algorithm:  [eqs. 38–41]
  #   1. C         <- if NULL: (1/T) t(X) %*% X
  #   2. Spcavec   <- eigenvalues of C  (descending)
  #   3. Swvec     <- singular values of W  (descending); pad to length m
  #   4. a_max     <- Spcavec %*% (Swvec^2)
  #      a_min     <- Spcavec %*% rev(Swvec^2)
  #      a_raw     <- sum(diag(t(W) %*% C %*% W))
  #      alpha_in  <- (a_raw - a_min) / (a_max - a_min)
  # --------------------------------------------------------
}
```

### `rrr_alignment_output(X, Y, W)`

> **Code note:** `main_figures.m` (fig 6) uses a different formula for alpha_out^raw:
> `muscom - muspop` (difference of means of cumulative variance distributions).
> This differs from `alignment_output.m` and `alignment.py`, which implement the paper's
> formula (eq. 44: dot product of eigenvalues and communicated variances).
> **Use the paper's formula (eq. 44), as implemented in `alignment_output.m`.**

```r
rrr_alignment_output <- function(X, Y, W) {
  # [alpha_out, comm_frac] = rrr_alignment_output(X, Y, W)
  # Compute output alignment index and communication fraction.
  #
  # alpha_out : scalar ∈ [0, 1]  (1 = communication concentrated in dominant output PCs)
  # comm_frac : scalar ∈ [0, 1]  (total communication variance / total output variance)
  #
  # X: matrix [T × m], centered input activity
  # Y: matrix [T × n], centered output activity
  # W: weight matrix [m × n]
  #
  # Algorithm:  [eqs. 42–47]
  #   1. Sigma_X    <- (1/T) t(X) %*% X
  #      Sigma_Y    <- cov(Y)
  #   2. PCA of Sigma_Y: g(upop, spopvec) %=% eigen(Sigma_Y)
  #   3. cov_com   <- W %*% Sigma_X %*% t(W)   [n × n]
  #      scomvec   <- diag(t(upop) %*% cov_com %*% upop)  [n]  communicated variance per output PC
  #   4. comm_frac <- sum(scomvec) / sum(spopvec)          [eq. 42]
  #   5. a_raw     <- sum(scomvec * spopvec)               [eq. 44]
  #   6. Compute a_max and a_min via rearrangement          [eqs. 45–46]
  #      alpha_out <- (a_raw - a_min) / (a_max - a_min)   [eq. 47]
  # --------------------------------------------------------
}
```

### `rrr_simulate(T, nx, ny, rank, sigma_noise, Sigma)`

```r
rrr_simulate <- function(T = 100, nx = 10, ny = 5, rank = 1, sigma_noise = 0.1, Sigma = NULL) {
  # [X, Y, U, V] = rrr_simulate(...)
  # Generate simulated data from the RRR generative model.
  #
  # X : matrix [T × nx]
  # Y : matrix [T × ny]
  # U : true input axes  [nx × rank]
  # V : true output axes [ny × rank]
  #
  # Algorithm:
  #   X  <- randn(T, nx); X <- scale(X, center=TRUE, scale=FALSE)
  #   U  <- randn(nx, rank); V <- randn(ny, rank)
  #   B  <- U %*% t(V)   [nx × ny]
  #   E  <- if Sigma NULL: sigma_noise * randn(T, ny)
  #          else: MASS::mvrnorm(T, mu=rep(0,ny), Sigma=Sigma)
  #   Y  <- X %*% B + E
  # --------------------------------------------------------
}
```

---

## 9. Test Strategy

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

```r
test_rrr_fit_rank <- function() {
  # matrix_rank(model$W) == rank  (up to machine epsilon)
}

test_rrr_v_orthonormal <- function() {
  # max(abs(t(model$V) %*% model$V - diag(rank))) < 1e-10
}

test_rrr_fit_reduces_to_ols_at_full_rank <- function() {
  # rrr_fit(X, Y, rank=min(m,n)-1, lambda=0)$W ≈ W_LS   (for full-rank X, Y)
}

test_rrr_r2_nonneg_on_train <- function() {
  # rrr_r2(X, Y, rrr_fit(X, Y, rank)) >= 0  (trivially; less than OLS R²)
}

test_rrr_fit_noniso_convergence <- function() {
  # Generate data with known non-spherical Sigma; verify W_fcRRR recovers U, V better than standard
}

test_alignment_input_extremes <- function() {
  # Construct W aligned exactly with top-r PCs of X → alpha_in ≈ 1
  # Construct W aligned with bottom-r PCs         → alpha_in ≈ 0
}

test_alignment_output_extremes <- function() {
  # Construct communication that drives top output PCs  → alpha_out ≈ 1
  # Construct communication that drives bottom PCs      → alpha_out ≈ 0
}

test_comm_fraction_bounds <- function() {
  # 0 <= comm_frac <= 1 on training data
}
```

---

## 10. Implementation Order

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

## 11. Coding Conventions

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

## 12. Design Principles

1. **Understanding before code.** Every function maps to specific equations in Wu & Pillow (2025). Write the equation number in a comment before the corresponding line.

2. **Test against Python reference on identical data.** Numerical agreement (max absolute error < 1e-6) required before any function is considered complete. Sign flips in U and V columns are acceptable.

3. **`rrr_fit` handles both standard and ridge RRR via `lambda`.** Lambda = 0 gives standard OLS. No separate function for ridge; this avoids divergence in logic.

4. **Full-covariance RRR is a separate function** (`rrr_fit_noniso`) because its iterative structure, non-orthogonal V, and dependency on `expm::sqrtm` represent a substantively different algorithm.

5. **Alignment metrics take W as input**, not a model object. This allows computing alignment for any weight matrix (not just from `rrr_fit`), e.g. from OLS.

6. **Cross-validation shuffles row indices**, not fold indices. Time structure is not assumed; if temporal autocorrelation is a concern, block CV should be used (caller's responsibility).

7. **Centering is the caller's responsibility.** `rrr_fit` and all metric functions receive already-centered matrices. This separates preprocessing from estimation.

---

## 13. Connection to Existing Projects

| Context | RRR role |
|---------|----------|
| dPCA/jPCA (this repo) | RRR identifies inter-region communication subspace; dPCA identifies within-region task subspaces; complementary questions |
| EEGMRI_RuleAction two-region EEG | X = region A activity (e.g. frontal), Y = region B activity (e.g. parietal); RRR finds communication subspace across rule conditions |
| EmbeddingRNN | Apply RRR to hidden layer pairs in RNN to ask whether conjunctive coding is reflected in low-dimensional inter-layer communication |

---

## 14. Open Questions

- **Matrix square root in `rrr_fit_noniso`:** `expm::sqrtm` is general but slow; for diagonal Sigma, `sqrt(diag(Sigma))` suffices. Add a diagonal-Sigma fast path?
- **Rank selection in small-T regime:** CV may favor rank = 1 spuriously when T < m. Should minimum rank = 2 be enforced for meaningful communication subspace?
- **Alignment metric symmetry:** Wu & Pillow define separate input and output alignment. Is there a joint metric that captures both simultaneously?
- **Time-lagged RRR:** For EEG data, input drives output with a lag. Optimal lag selection could be integrated into `rrr_cv_rank` as an additional hyperparameter axis.

---

## 15. References

- **Wu, B. & Pillow, J.W. (2025).** Reduced rank regression for neural communication: a tutorial for neuroscientists. *arXiv:2512.12467v1* ← primary reference for all mathematical foundations
- **Semedo, J.D. et al. (2019).** Cortical areas interact through a communication subspace. *Neuron* 102(1), 249–259. ← empirical application establishing communication subspace concept
- **Izenman, A.J. (1975).** Reduced-rank regression for the multivariate linear model. *Journal of Multivariate Analysis* 5(2), 248–264. ← original statistical paper [ref 1 in Wu & Pillow]
- **Wu & Pillow original code.** MATLAB and Python implementations in `original/RRR/` ← reference for numerical verification
- **noiseTools_lib.R** (this project). MCCA conversion from MATLAB → R. ← coding convention reference
