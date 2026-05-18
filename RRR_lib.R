# RRR library
# ------------------------------------------------------------------------------
# R implementation of Wu & Pillow (2025) Reduced Rank Regression
# MATLAB/Python originals: https://github.com/bichanw/RRR
#
# Core functions:
#   rrr_simulate         -- generate data from RRR generative model
#   rrr_fit              -- standard RRR with optional ridge regularization
#   rrr_transform        -- predict Y_hat = X W
#   rrr_r2               -- R² (fraction of Y variance explained)
#   rrr_cv_rank          -- cross-validated rank and lambda selection
#   rrr_fit_noniso       -- full-covariance RRR (iterative, non-spherical noise)
#   rrr_alignment_input  -- input alignment index alpha_in  [0, 1]
#   rrr_alignment_output -- output alignment index alpha_out [0, 1] + comm_frac
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

.mat_sqrt_psd <- function(C) {
  # Symmetric matrix square root for PSD C via eigendecomposition.
  # Avoids expm dependency; numerically stable for covariance matrices.
  e <- eigen(C, symmetric = TRUE)
  e$vectors %*% diag(sqrt(pmax(e$values, 0))) %*% t(e$vectors)
}

.mat_inv_sqrt_psd <- function(C, tol = 1e-10) {
  # Symmetric inverse square root for PSD C. Truncates near-zero eigenvalues.
  e <- eigen(C, symmetric = TRUE)
  e$vectors %*% diag(1 / sqrt(pmax(e$values, tol))) %*% t(e$vectors)
}


# ==============================================================================
# Core functions
# ==============================================================================

rrr_simulate <- function(T = 100, nx = 10, ny = 5, rank = 1,
                          sigma_noise = 0.1, Sigma = NULL) {
  # [sim] = rrr_simulate(T, nx, ny, rank, sigma_noise, Sigma)
  # Generate data from the RRR generative model: Y = X U V^T + E
  #
  # sim$X : matrix [T x nx], centered input activity
  # sim$Y : matrix [T x ny], noisy output activity
  # sim$U : true input axes  [nx x rank]
  # sim$V : true output axes [ny x rank]
  # sim$B : true weight matrix [nx x ny] = U %*% t(V)
  #
  # T          : number of time samples
  # nx         : number of input neurons
  # ny         : number of output neurons
  # rank       : communication rank
  # sigma_noise: noise standard deviation (isotropic; ignored if Sigma provided)
  # Sigma      : noise covariance [ny x ny] (optional; if NULL use sigma_noise * I)
  # --------------------------------------------------------
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


rrr_fit <- function(X, Y, rank, lambda = 0) {
  # [model] = rrr_fit(X, Y, rank, lambda)
  # Fit reduced rank regression with optional ridge regularization.
  #
  # model$U     : input axes  [m x r]
  # model$V     : output axes [n x r], V^T V = I (semi-orthogonal)
  # model$W     : weight matrix [m x n] = U %*% t(V)
  # model$W_ls  : ridge/OLS estimate [m x n]
  # model$rank  : r
  # model$lambda: lambda used
  #
  # X     : matrix [T x m], centered
  # Y     : matrix [T x n], centered
  # rank  : integer r, communication dimensionality
  # lambda: ridge penalty >= 0  (0 = standard OLS)
  #
  # Algorithm: Wu & Pillow (2025) eqs. 14-17
  # --------------------------------------------------------
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


rrr_transform <- function(X, model) {
  # [Y_hat] = rrr_transform(X, model)
  # Predict output from input using fitted RRR model.
  #
  # Y_hat: matrix [T x n]
  #
  # X    : matrix [T x m], centered (same centering as in rrr_fit)
  # model: output of rrr_fit or rrr_fit_noniso
  # --------------------------------------------------------
  X %*% model$W
}


rrr_r2 <- function(X, Y, model) {
  # [r2] = rrr_r2(X, Y, model)
  # Compute R²: fraction of Y variance explained by RRR prediction.
  #
  # r2: scalar (>= 0 on training data; can be negative on held-out data)
  #
  # X    : matrix [T x m], centered
  # Y    : matrix [T x n], centered
  # model: output of rrr_fit or rrr_fit_noniso
  # --------------------------------------------------------
  Y_hat <- X %*% model$W
  1 - sum((Y - Y_hat)^2) / sum(Y^2)
}


rrr_cv_rank <- function(X, Y, max_rank, lambda_grid = 0, n_folds = 10) {
  # [result] = rrr_cv_rank(X, Y, max_rank, lambda_grid, n_folds)
  # Select communication rank (and ridge lambda) by k-fold cross-validation.
  #
  # result$best_rank  : integer, best rank by mean held-out R²
  # result$best_lambda: numeric, best lambda
  # result$cv_r2      : matrix [max_rank x length(lambda_grid)], mean held-out R²
  #
  # X          : matrix [T x m], centered
  # Y          : matrix [T x n], centered
  # max_rank   : integer, maximum rank to evaluate
  # lambda_grid: numeric vector of lambda values (default = 0)
  # n_folds    : integer
  # --------------------------------------------------------
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


rrr_fit_noniso <- function(X, Y, rank, max_iter = 20, tol = 1e-6) {
  # [model] = rrr_fit_noniso(X, Y, rank, max_iter, tol)
  # Full-covariance RRR with iterative estimation of noise covariance Sigma.
  #
  # model$U     : input axes  [m x r]
  # model$V     : output axes [n x r]  (not semi-orthogonal in general)
  # model$W     : weight matrix [m x n] = U %*% t(V)
  # model$Sigma : estimated noise covariance [n x n]
  # model$rank  : r
  # model$n_iter: iterations until convergence
  #
  # X       : matrix [T x m], centered
  # Y       : matrix [T x n], centered
  # rank    : integer r
  # max_iter: maximum EM iterations (typically converges in < 10)
  # tol     : convergence threshold on max(abs(W_new - W_old))
  #
  # Algorithm: Wu & Pillow (2025) Section 5.2, eqs. 35-37
  # Coordinate ascent on Sigma: alternate W_fcRRR | Sigma and Sigma | W_fcRRR
  # --------------------------------------------------------
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

rrr_alignment_input <- function(X, W, r = NULL, C = NULL) {
  # [alpha_in] = rrr_alignment_input(X, W, r, C)
  # Compute input alignment index: how much communication aligns
  # with dominant input PCs.
  #
  # alpha_in: scalar in [0, 1]  (1 = maximally aligned with top input PCs)
  #
  # X: matrix [T x m], centered input activity
  # W: weight matrix [m x n]  (e.g. model$W from rrr_fit)
  # r: rank (optional; if NULL infer from ncol(W))
  # C: input covariance [m x m] (optional; if NULL compute cov(X))
  #
  # Reference: Wu & Pillow (2025) eqs. 38-41
  # --------------------------------------------------------
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


rrr_alignment_output <- function(X, Y, W) {
  # [result] = rrr_alignment_output(X, Y, W)
  # Compute output alignment index and communication fraction.
  #
  # result$alpha_out : scalar in [0, 1]  (1 = communication in dominant output PCs)
  # result$comm_frac : scalar in [0, 1]  (total comm variance / total output variance)
  #
  # X: matrix [T x m], centered input activity
  # Y: matrix [T x n], centered output activity
  # W: weight matrix [m x n]
  #
  # Note: uses the paper's formula (eq. 44), not the muscom-muspop variant
  # in main_figures.m (which differs; see ARCHITECTURE_RRR.md for details).
  #
  # Reference: Wu & Pillow (2025) eqs. 42-47
  # --------------------------------------------------------
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
