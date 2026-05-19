# visualizations/vis_rrr_test.R
# ------------------------------------------------------------------------------
# Visualize RRR_lib.R results using ggplot2 + patchwork
# Output: visualizations/rrr_fig1_cv_r2.png
#         visualizations/rrr_fig2_alignment_input.png
#         visualizations/rrr_fig3_alignment_output.png
# Run from project root: source("visualizations/vis_rrr_test.R")
# ------------------------------------------------------------------------------

library(ggplot2)
library(patchwork)
source("RRR_lib.R")

out_dir <- "visualizations"
dir.create(out_dir, showWarnings = FALSE)

# ==============================================================================
# Surrogate data
# ==============================================================================

set.seed(42)
sim   <- rrr_simulate(T = 300, nx = 10, ny = 8, rank = 2, sigma_noise = 0.3)
X     <- sim$X
Y     <- sim$Y
model <- rrr_fit(X, Y, rank = 2)

# ==============================================================================
# Figure 1: CV R² vs rank
# ==============================================================================

set.seed(7)
cv <- rrr_cv_rank(X, Y, max_rank = 6,
                  lambda_grid = c(0, 1, 10, 100), n_folds = 10)

fig1 <- rrr_plot_cv_r2(cv) +
  ggplot2::labs(title = "Figure 1 — Cross-validated R² vs rank",
                subtitle = sprintf("best rank = %d, best lambda = %g",
                                   cv$best_rank, cv$best_lambda))

ggsave(file.path(out_dir, "rrr_fig1_cv_r2.png"),
       fig1, width = 6, height = 4, dpi = 120)

# ==============================================================================
# Figure 2: Input alignment (3 scenarios)
# ==============================================================================

eig_X  <- eigen(cov(X), symmetric = TRUE)
U_pca  <- eig_X$vectors

# Three scenarios: aligned, anti-aligned, and fitted RRR weight
W_aligned <- U_pca[, 1:2] %*% matrix(rnorm(2 * 8), 2, 8)
W_antialign <- U_pca[, (ncol(U_pca) - 1):ncol(U_pca)] %*% matrix(rnorm(2 * 8), 2, 8)

scenarios <- list(
  "aligned (alpha~1)"    = W_aligned,
  "RRR fit"              = model$W,
  "anti-aligned (alpha~0)" = W_antialign
)

fig2_panels <- lapply(names(scenarios), function(nm) {
  rrr_plot_alignment_input(X, scenarios[[nm]]) +
    ggplot2::labs(title = nm)
})

fig2 <- patchwork::wrap_plots(fig2_panels, nrow = 1) +
  patchwork::plot_annotation(
    title = "Figure 2 — Input alignment: variance spectra by PC"
  )

ggsave(file.path(out_dir, "rrr_fig2_alignment_input.png"),
       fig2, width = 12, height = 4, dpi = 120)

# ==============================================================================
# Figure 3: Output alignment (3 scenarios)
# ==============================================================================

eig_Y   <- eigen(cov(Y), symmetric = TRUE)
U_y_pca <- eig_Y$vectors

# Three W scenarios targeting top, bottom, and random output PCs
W_top  <- matrix(rnorm(10), 10, 1) %*% t(U_y_pca[, 1, drop = FALSE])
W_bot  <- matrix(rnorm(10), 10, 1) %*% t(U_y_pca[, ncol(U_y_pca), drop = FALSE])
W_rnd  <- model$W

out_scenarios <- list(
  "top output PC"    = W_top,
  "RRR fit"          = W_rnd,
  "bottom output PC" = W_bot
)

fig3_panels <- lapply(names(out_scenarios), function(nm) {
  rrr_plot_alignment_output(X, Y, out_scenarios[[nm]]) +
    patchwork::plot_annotation(title = nm)
})

fig3 <- patchwork::wrap_plots(fig3_panels, ncol = 1) +
  patchwork::plot_annotation(
    title = "Figure 3 — Output alignment: variance spectra and scatter"
  )

ggsave(file.path(out_dir, "rrr_fig3_alignment_output.png"),
       fig3, width = 10, height = 9, dpi = 120)

# ==============================================================================
# Done
# ==============================================================================

cat("Done. Output written to visualizations/\n")
cat("  rrr_fig1_cv_r2.png\n")
cat("  rrr_fig2_alignment_input.png\n")
cat("  rrr_fig3_alignment_output.png\n")
