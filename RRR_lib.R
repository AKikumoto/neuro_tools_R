# RRR_lib.R
# ==============================================================================
# R implementation of Wu & Pillow (2025) Reduced Rank Regression
# MATLAB/Python originals: https://github.com/bichanw/RRR
#
# Functions:
#   rrr_simulate         -- generate data from the RRR generative model Y = X U V^T + E
#   rrr_fit              -- standard RRR: rank-r projection of OLS/ridge estimate
#   rrr_transform        -- predict Y_hat = X W
#   rrr_r2               -- R²: fraction of Y variance explained
#   rrr_cv_rank          -- cross-validated rank and lambda selection
#   rrr_fit_noniso       -- full-covariance RRR (coordinate ascent, non-spherical noise)
#   rrr_alignment_input  -- input alignment index alpha_in  in [0, 1]
#   rrr_alignment_output -- output alignment index alpha_out in [0, 1] + comm_frac
#
# Data convention:
#   X : matrix [T x m]   input-region activity, centered per column
#   Y : matrix [T x n]   output-region activity, centered per column
#   Centering is the caller's responsibility.
#
# Reference:
#   Wu B & Pillow JW (2025). Reduced rank regression for neural communication:
#   a tutorial for neuroscientists. arXiv:2512.12467v1.
# ==============================================================================


# ==============================================================================
# Dependencies
# ==============================================================================

## %=% parsing list output to multiple variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %-%, extendToMatch(), g()
# EX) g(a, b, c)  %=%  list("hello", 123, list("apples, oranges"))

'%=%' = function(l, r, ...) UseMethod('%=%')

'%=%.lbunch' = function(l, r, ...) {
  Envir = as.environment(-1)

  if (length(r) > length(l))
    warning("RHS has more args than LHS. Only first", length(l), "used.")

  if (length(l) > length(r))
    r <- extendToMatch(r, l)

  for (II in 1:length(l))
    do.call('<-', list(l[[II]], r[[II]]), envir=Envir)
}

extendToMatch <- function(source, destin) {
  s <- length(source)
  d <- length(destin)
  if (d==1 && s>1 && !is.null(as.numeric(destin))) d <- destin
  if (d - s > 0) source <- rep(source, ceiling(d/s))[1:d]
  return(source)
}

g = function(...) {
  List = as.list(substitute(list(...)))[-1L]
  class(List) = 'lbunch'
  return(List)
}


# ==============================================================================
# Internal helpers
# ==============================================================================

# =========================================================================
# .mat_sqrt_psd: SYMMETRIC MATRIX SQUARE ROOT FOR PSD MATRICES
# =========================================================================
# [Role]:
#   Compute C^{1/2} for a positive semidefinite (PSD) matrix C via
#   eigendecomposition: C = Q Lambda Q^T => C^{1/2} = Q Lambda^{1/2} Q^T.
#   Required by rrr_fit_noniso to whiten and un-whiten the cross-covariance
#   in the non-isotropic noise SVD step (eq. 36).
#
# [Inputs]:
#   C : matrix [n x n], symmetric PSD (e.g., noise covariance Sigma)
#
# [Outputs]:
#   C^{1/2} : matrix [n x n], symmetric PSD
#
# [Notes]:
#   - Avoids the expm package; eigendecomposition is stable for covariance matrices.
#   - Near-zero eigenvalues are clipped at 0 via pmax to prevent NaN in sqrt.
#
# [Reference]:
#   Wu & Pillow (2025) Sec. 5.2, eq. 36
# =========================================================================
.mat_sqrt_psd <- function(C) {
  e <- eigen(C, symmetric = TRUE)
  e$vectors %*% diag(sqrt(pmax(e$values, 0))) %*% t(e$vectors)
}

# =========================================================================
# .mat_inv_sqrt_psd: SYMMETRIC INVERSE SQUARE ROOT FOR PSD MATRICES
# =========================================================================
# [Role]:
#   Compute C^{-1/2} for a PSD matrix C, used to whiten the cross-covariance
#   before SVD in rrr_fit_noniso (eq. 36): transform to spherical noise space,
#   solve the isotropic RRR there, then un-whiten the output axes.
#
# [Inputs]:
#   C   : matrix [n x n], symmetric PSD
#   tol : numeric, eigenvalues below this are clamped to tol (default 1e-10)
#
# [Outputs]:
#   C^{-1/2} : matrix [n x n]
#
# [Notes]:
#   - Eigenvalues below tol are set to tol before inversion (regularization).
#   - If C is exactly singular the pseudo-inverse square root is returned.
#
# [Reference]:
#   Wu & Pillow (2025) Sec. 5.2, eq. 36
# =========================================================================
.mat_inv_sqrt_psd <- function(C, tol = 1e-10) {
  e <- eigen(C, symmetric = TRUE)
  e$vectors %*% diag(1 / sqrt(pmax(e$values, tol))) %*% t(e$vectors)
}


# ==============================================================================
# Core functions
# ==============================================================================

# =========================================================================
# rrr_simulate: GENERATE SYNTHETIC DATA FROM THE RRR GENERATIVE MODEL
# =========================================================================
# [Role]:
#   Simulate data from the rank-r linear-Gaussian model:
#     Y = X U V^T + E,    E ~ N(0, Sigma) row-wise
#   Used to test rank recovery, alignment metrics, and cross-validation.
#   The true weight matrix B = U V^T has rank exactly r (almost surely).
#
# [Inputs]:
#   T          : integer, number of time samples
#   nx         : integer, number of input neurons (m)
#   ny         : integer, number of output neurons (n)
#   rank       : integer, communication rank (r)
#   sigma_noise: numeric, isotropic noise std (ignored if Sigma is given)
#   Sigma      : matrix [ny x ny], noise covariance (optional; overrides sigma_noise)
#
# [Outputs]:
#   $X : matrix [T x nx], centered Gaussian input activity
#   $Y : matrix [T x ny], noisy output Y = X B + E
#   $U : matrix [nx x rank], true input axes
#   $V : matrix [ny x rank], true output axes
#   $B : matrix [nx x ny] = U V^T, true weight matrix
#
# [Algorithm]:
#   1. Draw X ~ N(0, I) [T x nx]; center columns.
#   2. Draw U ~ N(0, I) [nx x rank], V ~ N(0, I) [ny x rank]; B = U V^T.
#   3. Draw noise:
#        isotropic:   E[t, :] ~ N(0, sigma_noise^2 I)
#        correlated:  E ~ MASS::mvrnorm(T, 0, Sigma)
#   4. Return Y = X B + E.
#
# [Notes]:
#   - U, V are not orthonormal; B has exact numerical rank r.
#   - X is centered to match the caller's centering convention (data convention).
#
# [Reference]:
#   Wu & Pillow (2025) eqs. 3-5, Sec. 2.1
# =========================================================================
rrr_simulate <- function(T = 100, nx = 10, ny = 5, rank = 1,
                          sigma_noise = 0.1, Sigma = NULL) {
  X <- matrix(rnorm(T * nx), nrow = T, ncol = nx)
  X <- sweep(X, 2, colMeans(X), "-")   # center columns

  U <- matrix(rnorm(nx * rank), nrow = nx, ncol = rank)
  V <- matrix(rnorm(ny * rank), nrow = ny, ncol = rank)
  B <- U %*% t(V)

  if (is.null(Sigma)) {
    E <- matrix(rnorm(T * ny, sd = sigma_noise), nrow = T, ncol = ny)
  } else {
    E <- MASS::mvrnorm(T, mu = rep(0, ny), Sigma = Sigma)
  }

  Y <- X %*% B + E
  list(X = X, Y = Y, U = U, V = V, B = B)
}


# =========================================================================
# rrr_fit: STANDARD REDUCED RANK REGRESSION WITH OPTIONAL RIDGE
# =========================================================================
# [Role]:
#   Estimate the rank-r communication weight matrix W = U V^T that minimizes
#   ||Y - X W||_F^2 subject to rank(W) <= r, with optional ridge penalty.
#
#   Key difference from SVD(W_LS):
#     When cov(X) != sigma^2 I (non-spherical input), RRR and SVD(W_LS) differ.
#     RRR finds the subspace capturing the most Y variance reachable through X,
#     while SVD(W_LS) ignores the input distribution. Use RRR when X has
#     population structure (realistic neural data).
#     See Wu & Pillow (2025) Sec. 3.3 and ARCHITECTURE_RRR.md Sec. 6.
#
# [Inputs]:
#   X      : matrix [T x m], centered input activity
#   Y      : matrix [T x n], centered output activity
#   rank   : integer r, communication dimensionality
#   lambda : numeric >= 0, ridge penalty (0 = OLS; default 0)
#
# [Outputs]:
#   $U     : matrix [m x r], input axes (W = U V^T)
#   $V     : matrix [n x r], output axes (V^T V = I, semi-orthogonal by construction)
#   $W     : matrix [m x n] = U V^T, full weight matrix
#   $W_ls  : matrix [m x n], unrestricted OLS or ridge estimate
#   $rank  : integer r
#   $lambda: numeric lambda used
#
# [Algorithm]:
#   Step 1 (OLS/ridge estimate):
#     W_LS = (X^T X + lambda I)^{-1} X^T Y                       [eq. 14]
#   Step 2 (SVD for output subspace):
#     SVD of Y^T X W_LS  [n x n]; top r right singular vectors = V_r   [eq. 15]
#     V^T V = I by construction (columns of svd()$v are orthonormal).
#   Step 3 (project W_LS onto V_r):
#     U = W_LS V_r                                                [eq. 16]
#     W = U V_r^T                                                 [eq. 17]
#
# [Notes]:
#   - Falls back to MASS::ginv if rcond(X^T X + lambda I) < 1e-10.
#   - R's svd()$v columns are right singular vectors (unlike numpy which
#     returns V^T). This matches the mathematical V convention directly.
#   - V is semi-orthogonal (V^T V = I); U is not orthonormal in general.
#
# [Reference]:
#   Wu & Pillow (2025) eqs. 14-17, Sec. 3
# =========================================================================
rrr_fit <- function(X, Y, rank, lambda = 0) {
  m <- ncol(X)

  # Step 1: ridge/OLS estimate [eq. 14 / ridge]
  XX <- t(X) %*% X + lambda * diag(m)
  if (rcond(XX) < 1e-10) {
    W_ls <- MASS::ginv(XX) %*% (t(X) %*% Y)
  } else {
    W_ls <- solve(XX, t(X) %*% Y)    # [m x n]
  }

  # Step 2: SVD of Y^T X W_ls  (equivalent to W_ls^T X^T X W_ls, which is PSD)
  # Top r right singular vectors = output axes V_r  [eq. 15]
  sv <- svd(t(Y) %*% X %*% W_ls)     # n x n matrix; $v columns are right singular vectors
  V  <- sv$v[, seq_len(rank), drop = FALSE]   # [n x r], V^T V = I

  # Step 3: project W_ls onto V_r subspace  [eqs. 16-17]
  U  <- W_ls %*% V                    # [m x r], input axes
  W  <- U %*% t(V)                    # [m x n]

  list(U = U, V = V, W = W, W_ls = W_ls, rank = rank, lambda = lambda)
}


# =========================================================================
# rrr_transform: PREDICT OUTPUT FROM INPUT VIA FITTED RRR MODEL
# =========================================================================
# [Role]:
#   Compute Y_hat = X W using the fitted weight matrix. Applies to both
#   training data (same X as rrr_fit) and held-out test data.
#
# [Inputs]:
#   X     : matrix [T x m], centered (use training column means for test data)
#   model : list output of rrr_fit or rrr_fit_noniso (must contain $W)
#
# [Outputs]:
#   Y_hat : matrix [T x n]
#
# [Notes]:
#   - Contractual: exactly X %*% model$W, no hidden centering or scaling.
#   - If X is test data, center it using TRAINING column means before calling
#     (centering is the caller's responsibility; data convention).
#
# [Reference]:
#   Wu & Pillow (2025) eq. 13
# =========================================================================
rrr_transform <- function(X, model) {
  X %*% model$W
}


# =========================================================================
# rrr_r2: R² FRACTION OF OUTPUT VARIANCE EXPLAINED
# =========================================================================
# [Role]:
#   Compute R² = 1 - ||Y - X W||_F^2 / ||Y||_F^2.
#   On training data, R² >= 0 always (RRR dominates the zero predictor).
#   On held-out data, R² can be negative (model overfits noise dimensions).
#
# [Inputs]:
#   X     : matrix [T x m], centered
#   Y     : matrix [T x n], centered
#   model : list output of rrr_fit or rrr_fit_noniso (must contain $W)
#
# [Outputs]:
#   r2 : scalar, fraction of Y variance explained (>= 0 on training data)
#
# [Notes]:
#   - Denominator is ||Y||_F^2, not ||Y - mean(Y)||_F^2. Since Y is centered
#     (mean = 0 by data convention), these are identical.
#   - Used as the CV criterion in rrr_cv_rank (select rank maximizing held-out R²).
#
# [Reference]:
#   Wu & Pillow (2025) eq. 20
# =========================================================================
rrr_r2 <- function(X, Y, model) {
  Y_hat <- X %*% model$W
  1 - sum((Y - Y_hat)^2) / sum(Y^2)
}


# =========================================================================
# rrr_cv_rank: CROSS-VALIDATED RANK AND LAMBDA SELECTION
# =========================================================================
# [Role]:
#   Select communication rank r (and optionally ridge lambda) by k-fold CV:
#   fit on training folds, evaluate held-out R², pick the best (rank, lambda).
#
#   Motivation: training R² increases monotonically with rank; held-out R²
#   peaks at the true rank and then declines as noise dimensions are fitted.
#   CV identifies this peak.
#
# [Inputs]:
#   X          : matrix [T x m], centered
#   Y          : matrix [T x n], centered
#   max_rank   : integer, maximum rank to evaluate
#   lambda_grid: numeric vector of lambda values (default 0 = OLS only)
#   n_folds    : integer, number of CV folds (default 10)
#
# [Outputs]:
#   $best_rank  : integer, rank with highest mean held-out R²
#   $best_lambda: numeric, lambda with highest mean held-out R²
#   $cv_r2      : matrix [max_rank x length(lambda_grid)], mean held-out R²
#                 rows = ranks 1..max_rank, columns = lambda values
#
# [Algorithm]:
#   1. Assign each of T rows to one of n_folds folds (random, modular).
#   2. For each (rank, lambda): for each fold f, fit on rows NOT in f,
#      evaluate R² on rows IN f; store mean R² across folds.
#   3. Return (rank, lambda) argmax.
#
# [Notes]:
#   - Rank recovery requires T >> m for reliable results (Wu & Pillow Fig. 3A).
#   - With lambda_grid = 0 (default), only rank is selected.
#   - For small T, add ridge to the grid: lambda_grid = c(0, 1, 10, 100).
#
# [Reference]:
#   Wu & Pillow (2025) Sec. 4, Fig. 3
# =========================================================================
rrr_cv_rank <- function(X, Y, max_rank, lambda_grid = 0, n_folds = 10) {
  Tsamp    <- nrow(X)
  fold_ids <- sample(((seq_len(Tsamp) - 1) %% n_folds) + 1)

  cv_r2 <- matrix(NA_real_, nrow = max_rank, ncol = length(lambda_grid),
                  dimnames = list(paste0("rank", seq_len(max_rank)),
                                  paste0("lam", lambda_grid)))

  for (li in seq_along(lambda_grid)) {
    lam <- lambda_grid[li]
    for (r in seq_len(max_rank)) {
      fold_r2 <- numeric(n_folds)
      for (f in seq_len(n_folds)) {
        trn <- fold_ids != f
        tst <- fold_ids == f
        mod <- rrr_fit(X[trn, , drop = FALSE], Y[trn, , drop = FALSE],
                       rank = r, lambda = lam)
        fold_r2[f] <- rrr_r2(X[tst, , drop = FALSE], Y[tst, , drop = FALSE], mod)
      }
      cv_r2[r, li] <- mean(fold_r2)
    }
  }

  best_idx    <- which(cv_r2 == max(cv_r2, na.rm = TRUE), arr.ind = TRUE)[1, ]
  best_rank   <- as.integer(best_idx[1])
  best_lambda <- lambda_grid[best_idx[2]]

  list(best_rank = best_rank, best_lambda = best_lambda, cv_r2 = cv_r2)
}


# =========================================================================
# rrr_fit_noniso: FULL-COVARIANCE RRR WITH ITERATIVE NOISE ESTIMATION
# =========================================================================
# [Role]:
#   Estimate the rank-r weight matrix W under non-spherical Gaussian noise
#   E ~ N(0, Sigma). Isotropic rrr_fit is the special case Sigma = sigma^2 I.
#   Non-isotropic RRR is more efficient when output neurons have correlated
#   noise: the SVD in the whitened space finds the subspace maximizing
#   signal-to-noise along the directions of smallest noise, not total variance.
#
# [Inputs]:
#   X        : matrix [T x m], centered
#   Y        : matrix [T x n], centered
#   rank     : integer r
#   max_iter : integer, maximum coordinate-ascent iterations (default 20)
#   tol      : numeric, convergence threshold on max|W_new - W_old| (default 1e-6)
#
# [Outputs]:
#   $U     : matrix [m x r], input axes
#   $V     : matrix [n x r], output axes (NOT semi-orthogonal in general)
#   $W     : matrix [m x n] = U V^T
#   $Sigma : matrix [n x n], estimated noise covariance
#   $rank  : integer r
#   $n_iter: integer, actual iterations until convergence
#
# [Algorithm]:
#   Initialize: Sigma = I_n.
#   Iterate until convergence:
#     1. C^{1/2}   = .mat_sqrt_psd(Sigma)
#        C^{-1/2}  = .mat_inv_sqrt_psd(Sigma)
#     2. SVD of (C^{-1/2} Y^T X W_LS C^{-1/2})  ->  V_r  [eq. 36]
#     3. Un-whiten: V = C^{1/2} V_r                        [eq. 36]
#     4. U = (X^T X)^{-1} X^T Y Sigma^{-1} V              [eq. 37]
#     5. W = U V^T
#     6. Sigma = cov(Y - X W)
#   Stop when max|W - W_prev| < tol or max_iter reached.
#
# [Notes]:
#   - W_LS (OLS) is computed once and held fixed; Sigma whitens the SVD step
#     but does not update W_LS. See Wu & Pillow (2025) Sec. 5.2.
#   - n_iter == max_iter signals non-convergence; try smaller tol or more iterations.
#   - Typical convergence: < 10 iterations for neural data.
#
# [Reference]:
#   Wu & Pillow (2025) Sec. 5.2, eqs. 35-37
# =========================================================================
rrr_fit_noniso <- function(X, Y, rank, max_iter = 20, tol = 1e-6) {
  n <- ncol(Y)
  m <- ncol(X)

  # OLS estimate (fixed throughout; eqs. 36-37 reference W_LS, not W_fcRRR)
  W_ls <- solve(t(X) %*% X, t(X) %*% Y)    # [m x n]

  Sigma  <- diag(n)                          # initialize to isotropic  [Sec 5.2]
  W_prev <- matrix(0, nrow = m, ncol = n)
  U <- V <- W <- NULL
  n_iter <- 0L

  for (it in seq_len(max_iter)) {
    C_sqrt     <- .mat_sqrt_psd(Sigma)
    C_inv_sqrt <- .mat_inv_sqrt_psd(Sigma)

    # eq. 36: SVD of whitened cross-covariance → output axes in whitened space
    sv  <- svd(C_inv_sqrt %*% t(Y) %*% X %*% W_ls %*% C_inv_sqrt)
    V_r <- sv$v[, seq_len(rank), drop = FALSE]   # [n x r]
    V   <- C_sqrt %*% V_r                         # un-whiten  [eq. 36]

    # eq. 37: input axes
    Sigma_inv <- tryCatch(solve(Sigma), error = function(e) MASS::ginv(Sigma))
    U <- solve(t(X) %*% X, t(X) %*% Y %*% Sigma_inv %*% V)   # [m x r]
    W <- U %*% t(V)                                             # [m x n]

    # update Sigma from residuals
    Sigma <- cov(Y - X %*% W)

    dW <- max(abs(W - W_prev))
    W_prev <- W
    n_iter <- it
    if (it > 1 && dW < tol) break
  }

  list(U = U, V = V, W = W, Sigma = Sigma, rank = rank, n_iter = n_iter)
}


# ==============================================================================
# Alignment metrics
# ==============================================================================

# =========================================================================
# rrr_alignment_input: INPUT ALIGNMENT INDEX ALPHA_IN
# =========================================================================
# [Role]:
#   Measure how much the communication subspace W is aligned with the
#   dominant axes of input activity cov(X). alpha_in = 1 means communication
#   is carried entirely through the largest input PCs (maximum efficiency given
#   the input population structure); alpha_in = 0 means it is carried through
#   the smallest PCs (anti-aligned).
#
#   Neural interpretation: high alpha_in indicates that the input population
#   routes communication through its most active dimensions. Low alpha_in
#   means the input region uses its low-variance, "private" directions.
#
# [Inputs]:
#   X : matrix [T x m], centered input activity
#   W : matrix [m x n], communication weight matrix (e.g. model$W from rrr_fit)
#   r : integer (optional), rank; unused internally, kept for API consistency
#   C : matrix [m x m] (optional), input covariance; if NULL use cov(X)
#
# [Outputs]:
#   alpha_in : scalar in [0, 1]
#
# [Algorithm]:
#   Let lambda_1 >= ... >= lambda_m = eigenvalues of cov(X)
#   Let sigma_1  >= ... >= sigma_m  = singular values of W (padded with 0 to length m)
#
#   Communication variance:           a_raw = tr(W^T cov(X) W)        [eq. 38]
#   Maximum (rearrangement bound):    a_max = sum(lambda_i * sigma_i^2) [eq. 39]
#   Minimum (rearrangement bound):    a_min = sum(lambda_i * sigma_{m+1-i}^2) [eq. 40]
#
#   alpha_in = (a_raw - a_min) / (a_max - a_min)                      [eq. 41]
#
#   Bounds follow from the rearrangement inequality: sum of pairwise products
#   is maximized when both sequences share the same order, minimized when opposite.
#
# [Notes]:
#   - svd(C)$d gives eigenvalues of the PSD cov(X) equivalently to eigen()$values,
#     but svd is numerically preferred for near-singular covariance matrices.
#   - Singular values of W are padded to length m with zeros (rank-r W has m - r zeros).
#
# [Reference]:
#   Wu & Pillow (2025) eqs. 38-41, Sec. 6.1
# =========================================================================
rrr_alignment_input <- function(X, W, r = NULL, C = NULL) {
  m <- nrow(W)

  if (is.null(C)) C <- cov(X)

  # eigenvalues of Sigma_X, descending (= singular values since Sigma_X is PSD)
  Spcavec <- svd(C)$d                               # [m]

  # singular values of W, padded to length m  [eq. 39-40]
  Swvec <- svd(W)$d                                 # [min(m,n)]
  Swvec <- c(Swvec, rep(0, m - length(Swvec)))      # pad to [m]

  a_max <- sum(Spcavec * Swvec^2)                   # eq. 39: max comm variance
  a_min <- sum(Spcavec * rev(Swvec^2))              # eq. 40: min comm variance
  a_raw <- sum(diag(t(W) %*% C %*% W))             # eq. 38: actual comm variance

  (a_raw - a_min) / (a_max - a_min)                 # eq. 41
}


# =========================================================================
# rrr_alignment_output: OUTPUT ALIGNMENT INDEX ALPHA_OUT AND COMM_FRAC
# =========================================================================
# [Role]:
#   Measure how much the communicated variance (cov(X W)) is concentrated
#   in the dominant axes of output activity cov(Y). alpha_out = 1 means all
#   communicated variance lands in the top output PCs; alpha_out = 0 means
#   it lands in the weakest output PCs.
#
#   Also returns comm_frac = total communicated variance / total output variance,
#   measuring the overall amplitude of communication relative to the output
#   population.
#
#   Neural interpretation: high alpha_out means the input region "speaks"
#   into the dimensions that dominate output activity (strong functional
#   coupling). Low alpha_out means input drives marginal, low-variance
#   output dimensions.
#
# [Inputs]:
#   X : matrix [T x m], centered input activity
#   Y : matrix [T x n], centered output activity
#   W : matrix [m x n], communication weight matrix
#
# [Outputs]:
#   $alpha_out : scalar in [0, 1]
#   $comm_frac : scalar in [0, 1], total communicated var / total output var
#
# [Algorithm]:
#   Let mu_j, s_j = eigenvectors/values of cov(Y), s_1 >= ... >= s_n.
#   Communicated variance along output PC j:
#       gamma_j = mu_j^T cov(X W) mu_j                            [eq. 43]
#   Communication fraction:
#       CF = sum(gamma_j) / sum(s_j)                               [eq. 42]
#   Raw alignment:
#       a_raw = sum(s_j * gamma_j)                                 [eq. 44]
#   Bounds (greedy packing of totcom = sum(gamma_j)):
#       a_max: fill s_1, s_2, ... until totcom is placed           [eq. 45]
#       a_min: fill s_n, s_{n-1}, ... (smallest first)             [eq. 46]
#   alpha_out = (a_raw - a_min) / (a_max - a_min)                  [eq. 47]
#
# [Notes]:
#   - Uses the paper's dot-product formula (eq. 44), NOT the muscom-muspop
#     difference variant in main_figures.m, which differs numerically.
#     See ARCHITECTURE_RRR.md Sec. 10 for the discrepancy.
#   - Greedy packing: fill the largest eigenvalue bucket first until
#     totcom is exhausted; the remainder goes in the first bucket that
#     exceeds the running sum.
#
# [Reference]:
#   Wu & Pillow (2025) eqs. 42-47, Sec. 6.2
# =========================================================================
rrr_alignment_output <- function(X, Y, W) {
  sv_Y    <- svd(cov(Y))
  upop    <- sv_Y$u                                  # [n x n] output PC vectors
  spopvec <- sv_Y$d                                  # [n] output eigenvalues, descending
  spopcum <- cumsum(spopvec)

  # communicated variance along each output PC  [eq. 43]
  cov_pred <- cov(X %*% W)
  scomvec  <- diag(t(upop) %*% cov_pred %*% upop)   # [n]

  totcom   <- sum(scomvec)
  commfrac <- totcom / sum(spopvec)                  # eq. 42

  a_raw    <- sum(spopvec * scomvec)                 # eq. 44

  # max alignment: pack totcom into largest output PCs  [eq. 45]
  ii      <- which(spopcum > totcom + 1e-10)[1]
  scommax <- spopvec
  scommax[ii:length(scommax)] <- 0
  scommax[ii] <- totcom - sum(scommax)
  a_max   <- sum(spopvec * scommax)

  # min alignment: pack totcom into smallest output PCs  [eq. 46]
  spopvec_rev <- rev(spopvec)
  spopcum_rev <- cumsum(spopvec_rev)
  ii_rev      <- which(spopcum_rev > totcom + 1e-10)[1]
  scommin_rev <- spopvec_rev
  scommin_rev[ii_rev:length(scommin_rev)] <- 0
  scommin_rev[ii_rev] <- totcom - sum(scommin_rev)
  scommin <- rev(scommin_rev)
  a_min   <- sum(spopvec * scommin)

  alpha_out <- (a_raw - a_min) / (a_max - a_min)    # eq. 47

  list(alpha_out = alpha_out, comm_frac = commfrac)
}
