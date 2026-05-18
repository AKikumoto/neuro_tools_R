# notebook/test_RRR.R
# ==============================================================================
# Test suite for RRR_lib.R
#
# Unit tests:
#  1.  rrr_simulate: shapes correct
#  2.  rrr_simulate: Y ≈ X B up to noise (SNR > 5)
#  3.  rrr_fit: rank(W) == r
#  4.  rrr_fit: V is semi-orthogonal
#  5.  rrr_fit: R² non-negative on training data
#  6.  rrr_fit: ridge lambda reduces training R²
#  7.  rrr_transform: matches X %*% model$W exactly
#  8.  rrr_r2: non-negative on training data
#  9.  rrr_cv_rank: recovers true rank=2 in large-T regime
#  10. rrr_fit_noniso: output shapes correct
#  11. rrr_fit_noniso: converges within max_iter
#  12. rrr_alignment_input: in [0, 1]
#  13. rrr_alignment_input: ≈ 1 when W aligned with top input PC
#  14. rrr_alignment_input: ≈ 0 when W aligned with bottom input PC
#  15. rrr_alignment_output: alpha_out and comm_frac in [0, 1]
#  16. rrr_alignment_output: ≈ 1 when communication drives top output PC
#
# Python comparison (requires reticulate + numpy + scipy):
#  17. rrr_fit W matches svd_RRR w0 (lambda=0)
#  18. rrr_fit W matches svd_RRR w0 (lambda=100)
#  19. rrr_alignment_input matches Python alignment_input
#  20. rrr_alignment_output matches Python alignment_output
# ==============================================================================

library(testthat)

if (!exists("rrr_fit")) source("RRR_lib.R")


# ==============================================================================
# Shared toy data
# ==============================================================================

set.seed(42)
sim   <- rrr_simulate(T = 50, nx = 5, ny = 4, rank = 2, sigma_noise = 0.1)
X     <- sim$X
Y     <- sim$Y
model <- rrr_fit(X, Y, rank = 2)


# ==============================================================================
# Tests 1-2: rrr_simulate
# ==============================================================================

test_that("rrr_simulate returns correct shapes", {
  expect_equal(dim(sim$X), c(50L, 5L))
  expect_equal(dim(sim$Y), c(50L, 4L))
  expect_equal(dim(sim$U), c(5L, 2L))
  expect_equal(dim(sim$V), c(4L, 2L))
  expect_equal(dim(sim$B), c(5L, 4L))
})

test_that("rrr_simulate Y = X B + noise (SNR > 5)", {
  signal_var <- mean((X %*% sim$B)^2)
  noise_var  <- mean((Y - X %*% sim$B)^2)
  expect_gt(signal_var / noise_var, 5)
})


# ==============================================================================
# Tests 3-5: rrr_fit standard RRR
# ==============================================================================

test_that("rrr_fit W has numerical rank == r", {
  W_rank <- sum(svd(model$W)$d > 1e-8)
  expect_equal(W_rank, 2L)
})

test_that("rrr_fit V is semi-orthogonal (V^T V = I)", {
  err <- max(abs(t(model$V) %*% model$V - diag(2)))
  expect_lt(err, 1e-10)
})

test_that("rrr_fit R² is non-negative on training data", {
  expect_gte(rrr_r2(X, Y, model), 0)
})


# ==============================================================================
# Test 6: ridge lambda reduces training R²
# ==============================================================================

test_that("ridge lambda reduces training R² relative to lambda=0", {
  model_ridge <- rrr_fit(X, Y, rank = 2, lambda = 100)
  expect_lte(rrr_r2(X, Y, model_ridge), rrr_r2(X, Y, model) + 1e-12)
})


# ==============================================================================
# Tests 7-8: rrr_transform and rrr_r2
# ==============================================================================

test_that("rrr_transform produces X %*% model$W", {
  expect_equal(rrr_transform(X, model), X %*% model$W)
})

test_that("rrr_r2 is non-negative on training data", {
  expect_gte(rrr_r2(X, Y, model), 0)
})


# ==============================================================================
# Test 9: rrr_cv_rank rank recovery
# Note: probabilistic; set.seed ensures repeatability
# ==============================================================================

test_that("rrr_cv_rank recovers true rank=2 in large-T regime", {
  set.seed(99)
  sim_big <- rrr_simulate(T = 500, nx = 5, ny = 4, rank = 2, sigma_noise = 0.1)
  result  <- rrr_cv_rank(sim_big$X, sim_big$Y, max_rank = 4, n_folds = 5)
  expect_equal(result$best_rank, 2L)
})


# ==============================================================================
# Tests 10-11: rrr_fit_noniso
# ==============================================================================

test_that("rrr_fit_noniso returns correct shapes", {
  m_ni <- rrr_fit_noniso(X, Y, rank = 2)
  expect_equal(dim(m_ni$U),     c(5L, 2L))
  expect_equal(dim(m_ni$V),     c(4L, 2L))
  expect_equal(dim(m_ni$W),     c(5L, 4L))
  expect_equal(dim(m_ni$Sigma), c(4L, 4L))
})

test_that("rrr_fit_noniso converges within max_iter", {
  m_ni <- rrr_fit_noniso(X, Y, rank = 2, max_iter = 20, tol = 1e-6)
  expect_lte(m_ni$n_iter, 20L)
})


# ==============================================================================
# Tests 12-14: rrr_alignment_input
# Construction: W = u_top %*% matrix(1, 1, ny) has its left singular vector
# equal to u_top, so communication variance is entirely in the top (bottom)
# input PC direction, achieving alpha_in = 1 (0) exactly.
# ==============================================================================

eig_X <- eigen(cov(X), symmetric = TRUE)

test_that("rrr_alignment_input is in [0, 1]", {
  alpha <- rrr_alignment_input(X, model$W)
  expect_gte(alpha, 0)
  expect_lte(alpha, 1 + 1e-10)
})

test_that("rrr_alignment_input == 1 when W is aligned with top input PC", {
  u_top     <- eig_X$vectors[, 1, drop = FALSE]       # [5 x 1]
  W_aligned <- u_top %*% matrix(1, 1, 4)              # rank-1 [5 x 4]
  expect_gt(rrr_alignment_input(X, W_aligned), 1 - 1e-10)
})

test_that("rrr_alignment_input == 0 when W is aligned with bottom input PC", {
  u_bot        <- eig_X$vectors[, 5, drop = FALSE]    # [5 x 1]
  W_antialigned <- u_bot %*% matrix(1, 1, 4)
  expect_lt(rrr_alignment_input(X, W_antialigned), 1e-10)
})


# ==============================================================================
# Tests 15-16: rrr_alignment_output
# Construction: W = v_X %*% t(u_Y_top) maps input activity through one
# direction and places it entirely in the top output PC, so alpha_out = 1.
# ==============================================================================

eig_Y <- eigen(cov(Y), symmetric = TRUE)

test_that("rrr_alignment_output alpha_out and comm_frac are in [0, 1]", {
  res <- rrr_alignment_output(X, Y, model$W)
  expect_gte(res$alpha_out, 0)
  expect_lte(res$alpha_out, 1 + 1e-10)
  expect_gte(res$comm_frac, 0)
  expect_lte(res$comm_frac, 1 + 1e-10)
})

test_that("rrr_alignment_output == 1 when communication drives top output PC", {
  v_X     <- eig_X$vectors[, 1, drop = FALSE]         # [5 x 1] input direction
  u_Y_top <- eig_Y$vectors[, 1, drop = FALSE]         # [4 x 1] top output PC
  W_top   <- v_X %*% t(u_Y_top)                       # [5 x 4], rank 1
  res     <- rrr_alignment_output(X, Y, W_top)
  expect_gt(res$alpha_out, 1 - 1e-10)
})


# ==============================================================================
# Python comparison via reticulate
# Requires: reticulate, numpy, scipy
# ==============================================================================

if (requireNamespace("reticulate", quietly = TRUE)) {
  py_ok <- tryCatch({
    reticulate::import("numpy")
    reticulate::import("scipy")
    TRUE
  }, error = function(e) FALSE)

  if (py_ok) {
    py_path    <- file.path("original", "RRR", "python")
    fitting_py  <- reticulate::import_from_path("fitting",   path = py_path)
    align_py    <- reticulate::import_from_path("alignment", path = py_path)

    X_py <- reticulate::r_to_py(X)
    Y_py <- reticulate::r_to_py(Y)
    W_py_mat <- reticulate::r_to_py(model$W)

    test_that("rrr_fit W matches svd_RRR w0 (lambda=0)", {
      py_out <- fitting_py$svd_RRR(X_py, Y_py, 2L, lambda_ = 0.0)
      W_py   <- reticulate::py_to_r(py_out[[1]])
      expect_lt(max(abs(model$W - W_py)), 1e-8)
    })

    test_that("rrr_fit W matches svd_RRR w0 (lambda=100)", {
      model_r <- rrr_fit(X, Y, rank = 2, lambda = 100)
      py_out  <- fitting_py$svd_RRR(X_py, Y_py, 2L, lambda_ = 100.0)
      W_py    <- reticulate::py_to_r(py_out[[1]])
      expect_lt(max(abs(model_r$W - W_py)), 1e-8)
    })

    test_that("rrr_alignment_input matches Python alignment_input", {
      py_res   <- align_py$alignment_input(X_py, W_py_mat)
      alpha_py <- reticulate::py_to_r(py_res[[1]])
      alpha_r  <- rrr_alignment_input(X, model$W)
      expect_lt(abs(alpha_r - alpha_py), 1e-8)
    })

    test_that("rrr_alignment_output matches Python alignment_output", {
      py_res    <- align_py$alignment_output(X_py, Y_py, W_py_mat)
      alpha_py  <- reticulate::py_to_r(py_res[[1]])
      cf_py     <- reticulate::py_to_r(py_res[[2]])
      res_r     <- rrr_alignment_output(X, Y, model$W)
      expect_lt(abs(res_r$alpha_out - alpha_py), 1e-8)
      expect_lt(abs(res_r$comm_frac - cf_py),    1e-8)
    })

  } else {
    message("numpy/scipy not available; skipping Python comparison tests (17-20)")
  }
} else {
  message("reticulate not available; skipping Python comparison tests (17-20)")
}
