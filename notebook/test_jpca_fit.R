# tests/test_jpca_fit.R
# ------------------------------------------------------------------------------
# Unit tests for jpca_fit(), jpca_transform(), jpca_rotation_strength()
# Run: source("tests/test_jpca_fit.R")  or  testthat::test_file("tests/test_jpca_fit.R")
# ------------------------------------------------------------------------------

library(testthat)
source("jPCA_lib.R")


# ==============================================================================
# Surrogate data helpers
# ==============================================================================

# Pure rotation surrogate: Z(t) = R(ωt) Z(0) per condition
# Each condition starts from a different point on the unit circle
# noise_sd > 0 ensures full-rank data (avoids singular XtX in jpca_fit)
make_rotation_data <- function(N = 6, T = 50, C = 4, omega = 0.2,
                               noise_sd = 0.05, seed = 42) {
  set.seed(seed)
  t_seq  <- seq(0, T - 1)
  X_list <- vector("list", C)
  # Fixed random mixing matrix (same across conditions)
  A <- matrix(rnorm(N * 2), nrow = N)
  for (c in 1:C) {
    theta0 <- (c - 1) * 2 * pi / C
    # 2D rotation embedded in N dimensions
    X_2d <- rbind(cos(omega * t_seq + theta0),
                  sin(omega * t_seq + theta0))
    noise <- matrix(rnorm(N * T, sd = noise_sd), nrow = N)
    X_list[[c]] <- A %*% X_2d + noise  # [N × T]
  }
  return(X_list)
}

# Random data surrogate (no rotation structure)
make_random_data <- function(N = 6, T = 50, C = 4, seed = 99) {
  set.seed(seed)
  lapply(1:C, function(c) matrix(rnorm(N * T), nrow = N))
}


# ==============================================================================
# Test suite
# ==============================================================================

# ── Test 1: M_skew is skew-symmetric ──────────────────────────────────────────
test_that("M_skew is skew-symmetric", {
  X_list <- make_rotation_data()
  model  <- jpca_fit(X_list, n_pcs = 2)
  expect_lt(max(abs(model$M_skew + t(model$M_skew))), 1e-10)
})


# ── Test 2: Eigenvalues of M_skew are purely imaginary ────────────────────────
test_that("M_skew eigenvalues are purely imaginary", {
  X_list   <- make_rotation_data()
  model    <- jpca_fit(X_list, n_pcs = 2)
  eig_vals <- eigen(model$M_skew)$values
  expect_lt(max(abs(Re(eig_vals))), 1e-10)
})

# ── Test 3: jPC1 ⊥ jPC2 ───────────────────────────────────────────────────────
test_that("jPC1 is perpendicular to jPC2", {
  X_list <- make_rotation_data()
  model  <- jpca_fit(X_list, n_pcs = 2)
  dot12  <- sum(model$W[1, ] * model$W[2, ])
  expect_lt(abs(dot12), 1e-10)
})

# ── Test 4: jPC1 and jPC2 are unit vectors ────────────────────────────────────
test_that("jPC1 and jPC2 are unit vectors", {
  X_list <- make_rotation_data()
  model  <- jpca_fit(X_list, n_pcs = 2)
  expect_equal(sqrt(sum(model$W[1, ]^2)), 1, tolerance = 1e-10)
  expect_equal(sqrt(sum(model$W[2, ]^2)), 1, tolerance = 1e-10)
})

# ── Test 5: R2_skew ≤ R2_unrestr ──────────────────────────────────────────────
test_that("R2_skew <= R2_unrestr", {
  X_list <- make_rotation_data()
  model  <- jpca_fit(X_list, n_pcs = 2)
  expect_lte(model$R2_skew, model$R2_unrestr + 1e-10)
})

# ── Test 6: Pure rotation → R2_ratio > 0.85 ───────────────────────────────────
test_that("Pure rotation data yields R2_ratio > 0.85", {
  X_list <- make_rotation_data()
  model  <- jpca_fit(X_list, n_pcs = 2)
  proj   <- jpca_transform(X_list, model)
  result <- jpca_rotation_strength(proj, model)
  expect_gt(result$R2_ratio, 0.85)
})

# ── Test 7: Pure rotation → peak angle ≈ π/2 ─────────────────────────────────
test_that("Pure rotation data yields peak angle near pi/2", {
  X_list <- make_rotation_data()
  model  <- jpca_fit(X_list, n_pcs = 2)
  proj   <- jpca_transform(X_list, model)
  result <- jpca_rotation_strength(proj, model)
  expect_lt(abs(result$peak - pi / 2), 0.3)
})

# ── Test 8: Random data → R2_ratio < rotation data ───────────────────────────
test_that("Random data yields lower R2_ratio than pure rotation", {
  X_rot <- make_rotation_data()
  m_rot <- jpca_fit(X_rot, n_pcs = 2)
  r_rot <- jpca_rotation_strength(jpca_transform(X_rot, m_rot), m_rot)

  X_rnd <- make_random_data()
  m_rnd <- jpca_fit(X_rnd, n_pcs = 2)
  r_rnd <- jpca_rotation_strength(jpca_transform(X_rnd, m_rnd), m_rnd)

  expect_lt(r_rnd$R2_ratio, r_rot$R2_ratio)
})

# ── Test 9: Input validation ───────────────────────────────────────────────────
test_that("Non-list input raises error", {
  expect_error(jpca_fit(matrix(1:12, 3, 4)))
})

# ── Test 10: jpca_transform output shapes ─────────────────────────────────────
test_that("jpca_transform returns correct output shapes", {
  X_list <- make_rotation_data()
  model  <- jpca_fit(X_list, n_pcs = 2)
  proj   <- jpca_transform(X_list, model)
  C <- length(X_list)
  T <- ncol(X_list[[1]])

  expect_equal(nrow(proj$proj), 2)
  expect_equal(ncol(proj$proj), C * T)
  expect_length(proj$proj_list, C)
  expect_true(all(sapply(proj$proj_list,
                         function(m) nrow(m) == 2 && ncol(m) == T)))
})
