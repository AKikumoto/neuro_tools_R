# tests/test_demixedPCA.R
# ==============================================================================
# Test suite for demixedPCA_lib.R
#
# Mirrors the dPCA_demo.ipynb (Kobak et al. 2016) in R:
#   - Surrogate data: N=100 neurons, S=6 stimuli, T=250 time-points
#   - Two latent factors: zt (time modulation) and zs (stimulus modulation)
#   - Noise level: 0.2
#
# Tests:
#   1.  dpca_get_marginalizations returns all 2^K - 1 subsets in order
#   2.  Marginalizations sum to centered X (partition of variance)
#   3.  Each marginalization is orthogonal to the others
#   4.  dpca_fit P and D have correct shapes
#   5.  dpca_transform Z has correct shapes
#   6.  Explained variance is non-negative and ordered
#   7.  1st component of Z["s"] captures stimulus variance
#   8.  1st component of Z["t"] captures time variance
#   9.  Z["st"] 1st component captures interaction (less than s or t)
#   10. dpca_inverse_transform reconstructs to [N, S, T]
#   11. Reconstruction from all marginals approximates original data
#   12. 3-label case: get_marginalizations for "stc" has 7 entries
#   13. Marginalization with regularization: P/D shapes unchanged
#   14. n_components as named list
#   15. Explained variance of 1st "s" component > 1st "st" component
# ==============================================================================

library(testthat)

# dpca_fit should already be in environment (sourced by runner).
# Fallback for interactive use from project root:
if (!exists("dpca_fit")) source("demixedPCA_lib.R")


# ==============================================================================
# Shared surrogate data (mirrors dPCA_demo.ipynb)
# ==============================================================================
make_demo_data <- function(N = 100, T = 250, S = 6, noise = 0.2, seed = 42) {
  set.seed(seed)
  zt <- seq(0, T - 1) / T            # time factor
  zs <- seq(0, S - 1) / S            # stimulus factor

  X <- array(0, dim = c(N, S, T))
  for (n in seq_len(N)) {
    a_t <- rnorm(1)
    a_s <- rnorm(1)
    for (s in seq_len(S))
      for (t in seq_len(T))
        X[n, s, t] <- a_t * zt[t] + a_s * zs[s]
  }
  X + noise * array(rnorm(N * S * T), dim = c(N, S, T))
}

X <- make_demo_data()
N <- 100; S <- 6; T_len <- 250


# ==============================================================================
# Test 1: dpca_get_marginalizations structure
# ==============================================================================
test_that("get_marginalizations returns 2^K - 1 entries for K=2", {
  m <- dpca_get_marginalizations("st")
  expect_equal(length(m), 3L)
  expect_equal(names(m), c("s", "t", "st"))
  expect_equal(m[["s"]],  0L)
  expect_equal(m[["t"]],  1L)
  expect_equal(m[["st"]], c(0L, 1L))
})

# ==============================================================================
# Test 2: Marginalizations sum to centered X
# ==============================================================================
test_that("sum of all marginalizations equals centered X", {
  mXs   <- dpca_marginalize(X, "st")
  Xflat <- matrix(X, nrow = N)
  Xc    <- Xflat - rowMeans(Xflat)
  recon <- Reduce("+", mXs)
  expect_lt(max(abs(recon - Xc)), 1e-10)
})

# ==============================================================================
# Test 3: Marginalizations are mutually orthogonal
# ==============================================================================
test_that("marginalizations are mutually orthogonal (zero Frobenius inner product)", {
  mXs <- dpca_marginalize(X, "st")
  keys <- names(mXs)
  for (i in seq_along(keys)) {
    for (j in seq_along(keys)) {
      if (i >= j) next
      ip <- sum(mXs[[keys[i]]] * mXs[[keys[j]]])   # Frobenius inner product
      expect_lt(abs(ip), 1e-6,
                label = paste("inner product of", keys[i], "and", keys[j]))
    }
  }
})

# ==============================================================================
# Test 4: dpca_fit P and D shapes
# ==============================================================================
test_that("dpca_fit returns P and D with shapes [N x k] for each marginalization", {
  k     <- 3L
  model <- dpca_fit(X, "st", n_components = k)

  expect_equal(names(model$P), c("s", "t", "st"))
  expect_equal(names(model$D), c("s", "t", "st"))

  for (key in c("s", "t", "st")) {
    expect_equal(dim(model$P[[key]]), c(N, k))
    expect_equal(dim(model$D[[key]]), c(N, k))
  }
})

# ==============================================================================
# Test 5: dpca_transform output shapes
# ==============================================================================
test_that("dpca_transform returns arrays of shape [k, S, T]", {
  k     <- 3L
  model <- dpca_fit(X, "st", n_components = k)
  Z     <- dpca_transform(X, model)

  for (key in c("s", "t", "st")) {
    expect_equal(dim(Z[[key]]), c(k, S, T_len))
  }
})

# ==============================================================================
# Test 6: Explained variance is non-negative and ordered within each marginal
# ==============================================================================
test_that("explained variance ratios are non-negative and sorted descending", {
  model <- dpca_fit(X, "st", n_components = 5L)
  Z     <- dpca_transform(X, model)
  ev    <- attr(Z, "explained_variance_ratio")

  for (key in names(ev)) {
    expect_true(all(ev[[key]] >= -1e-10),
                label = paste("EV non-negative for", key))
    expect_true(all(diff(ev[[key]]) <= 1e-10),
                label = paste("EV sorted descending for", key))
  }
})

# ==============================================================================
# Test 7: 1st "s" component is correlated with the stimulus factor
# ==============================================================================
test_that("1st stimulus component Z[s] correlates with stimulus index", {
  model  <- dpca_fit(X, "st", n_components = 3L)
  Z      <- dpca_transform(X, model)
  # Z[["s"]][1, s, t]: should vary with s more than t
  # Compute variance over s and t for the 1st component
  z1     <- Z[["s"]][1, , ]           # [S, T]
  var_s  <- var(rowMeans(z1))         # variance over stimulus means
  var_t  <- var(colMeans(z1))         # variance over time means
  # for pure stimulus data, var_s should dominate
  expect_gt(var_s, var_t * 0.5)
})

# ==============================================================================
# Test 8: 1st "t" component is correlated with the time factor
# ==============================================================================
test_that("1st time component Z[t] correlates with time index", {
  model   <- dpca_fit(X, "st", n_components = 3L)
  Z       <- dpca_transform(X, model)
  z1      <- Z[["t"]][1, , ]          # [S, T]
  var_s   <- var(rowMeans(z1))
  var_t   <- var(colMeans(z1))
  expect_gt(var_t, var_s * 0.5)
})

# ==============================================================================
# Test 9: explained variance: s and t components capture most variance
# ==============================================================================
test_that("sum of s and t explained variance exceeds st explained variance", {
  model <- dpca_fit(X, "st", n_components = 3L)
  Z     <- dpca_transform(X, model)
  ev    <- attr(Z, "explained_variance_ratio")

  total_st <- sum(ev[["st"]])
  total_s  <- sum(ev[["s"]])
  total_t  <- sum(ev[["t"]])
  # the interaction term should be smaller than pure marginals for structured data
  expect_gt(total_s + total_t, total_st)
})

# ==============================================================================
# Test 10: dpca_inverse_transform returns [N, S, T]
# ==============================================================================
test_that("inverse_transform returns array [N, S, T]", {
  model <- dpca_fit(X, "st", n_components = 3L)
  Z     <- dpca_transform(X, model)
  Xrec  <- dpca_inverse_transform(Z[["s"]], model, "s")
  expect_equal(dim(Xrec), c(N, S, T_len))
})

# ==============================================================================
# Test 11: Sum of reconstructions from all marginals approximates centered X
# ==============================================================================
test_that("sum of all marginal reconstructions approximates centered X", {
  model   <- dpca_fit(X, "st", n_components = 6L)
  Z       <- dpca_transform(X, model)
  Xc_flat <- matrix(X, nrow = N)
  Xc_flat <- Xc_flat - rowMeans(Xc_flat)

  Xrec_sum <- array(0, dim = c(N, S, T_len))
  for (key in names(model$P)) {
    Xrec_sum <- Xrec_sum + dpca_inverse_transform(Z[[key]], model, key)
  }
  # with enough components, reconstruction should be reasonable
  rel_err <- sum((matrix(Xrec_sum, N) - Xc_flat)^2) / sum(Xc_flat^2)
  expect_lt(rel_err, 0.5)
})

# ==============================================================================
# Test 12: Three-label case "stc" produces 7 marginalizations
# ==============================================================================
test_that("get_marginalizations('stc') returns 7 entries", {
  m <- dpca_get_marginalizations("stc")
  expect_equal(length(m), 7L)
  expect_equal(names(m), c("s", "t", "c", "st", "sc", "tc", "stc"))
})

# ==============================================================================
# Test 13: Regularization does not change P/D shapes
# ==============================================================================
test_that("regularized dpca_fit still returns correct shapes", {
  k     <- 3L
  model <- dpca_fit(X, "st", n_components = k, regularizer = 0.01)
  for (key in c("s", "t", "st")) {
    expect_equal(dim(model$P[[key]]), c(N, k))
    expect_equal(dim(model$D[[key]]), c(N, k))
  }
})

# ==============================================================================
# Test 14: n_components as named list
# ==============================================================================
test_that("n_components as named list sets different k per marginalization", {
  nc    <- list(s = 2L, t = 4L, st = 1L)
  model <- dpca_fit(X, "st", n_components = nc)
  expect_equal(ncol(model$P[["s"]]),  2L)
  expect_equal(ncol(model$P[["t"]]),  4L)
  expect_equal(ncol(model$P[["st"]]), 1L)
})

# ==============================================================================
# Test 15: 1st explained variance of "s" > 1st of "st" for structured data
# ==============================================================================
test_that("stimulus component explains more variance than interaction for structured data", {
  model <- dpca_fit(X, "st", n_components = 3L)
  Z     <- dpca_transform(X, model)
  ev    <- attr(Z, "explained_variance_ratio")
  expect_gt(ev[["s"]][1], ev[["st"]][1])
})
