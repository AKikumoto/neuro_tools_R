# visualizations/vis_jpca_test.R
# ------------------------------------------------------------------------------
# Visualize jpca_fit() test results using ggplot2 + patchwork
# Output: visualizations/jpca_test_summary.md  +  visualizations/fig_*.png
# Run from project root: source("visualizations/vis_jpca_test.R")
# ------------------------------------------------------------------------------

library(ggplot2)
source("jPCA_lib.R")

out_dir <- "visualizations"
dir.create(out_dir, showWarnings = FALSE)

# ==============================================================================
# Surrogate data  (same seeds as test_jpca_fit.R)
# ==============================================================================

make_rotation_data <- function(N = 6, T = 50, C = 4, omega = 0.2,
                               noise_sd = 0.05, seed = 42) {
  set.seed(seed)
  t_seq <- seq(0, T - 1)
  A <- matrix(rnorm(N * 2), nrow = N)
  lapply(1:C, function(c) {
    theta0 <- (c - 1) * 2 * pi / C
    X_2d   <- rbind(cos(omega * t_seq + theta0), sin(omega * t_seq + theta0))
    A %*% X_2d + matrix(rnorm(N * T, sd = noise_sd), nrow = N)
  })
}

make_random_data <- function(N = 6, T = 50, C = 4, seed = 99) {
  set.seed(seed)
  lapply(1:C, function(c) matrix(rnorm(N * T), nrow = N))
}

X_rot <- make_rotation_data()
X_rnd <- make_random_data()

m_rot <- jpca_fit(X_rot, n_pcs = 2)
m_rnd <- jpca_fit(X_rnd, n_pcs = 2)

p_rot <- jpca_transform(X_rot, m_rot)
p_rnd <- jpca_transform(X_rnd, m_rnd)

r_rot <- jpca_rotation_strength(p_rot, m_rot)
r_rnd <- jpca_rotation_strength(p_rnd, m_rnd)

# ==============================================================================
# Helper: proj_list -> tidy data frame
# ==============================================================================

proj_to_df <- function(proj_list, label) {
  C <- length(proj_list)
  T <- ncol(proj_list[[1]])
  do.call(rbind, lapply(seq_len(C), function(c) {
    m <- proj_list[[c]]
    data.frame(
      jPC1      = m[1, ],
      jPC2      = m[2, ],
      timepoint = seq_len(T),
      condition = factor(paste0("Cond ", c)),
      data_type = label
    )
  }))
}

df_traj <- rbind(
  proj_to_df(p_rot$proj_list, "Pure rotation"),
  proj_to_df(p_rnd$proj_list, "Random")
)
df_traj$data_type <- factor(df_traj$data_type,
                             levels = c("Pure rotation", "Random"))

# ==============================================================================
# Figure 1: jPCA plane trajectories
# ==============================================================================

r2_labels <- c(
  "Pure rotation" = sprintf("R\u00b2_ratio = %.3f", r_rot$R2_ratio),
  "Random"        = sprintf("R\u00b2_ratio = %.3f", r_rnd$R2_ratio)
)

fig1 <- ggplot(df_traj, aes(x = jPC1, y = jPC2, colour = condition)) +
  geom_path(linewidth = 0.8, alpha = 0.85) +
  geom_point(data = df_traj[df_traj$timepoint == 1, ], size = 2.5) +
  geom_hline(yintercept = 0, colour = "grey70", linetype = "dashed",
             linewidth = 0.4) +
  geom_vline(xintercept = 0, colour = "grey70", linetype = "dashed",
             linewidth = 0.4) +
  facet_wrap(~data_type, scales = "free",
             labeller = labeller(data_type = r2_labels)) +
  labs(title = "Figure 1 \u2014 jPCA plane trajectories",
       subtitle = "Filled circle = t = 0 (start)",
       x = "jPC1", y = "jPC2", colour = NULL) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 10),
        aspect.ratio = 1)

ggsave(file.path(out_dir, "fig1_trajectories.png"),
       fig1, width = 9, height = 4.5, dpi = 120)

# ==============================================================================
# Figure 2: Angle distributions
# ==============================================================================

df_angles <- rbind(
  data.frame(angle = r_rot$angles, data_type = "Pure rotation"),
  data.frame(angle = r_rnd$angles, data_type = "Random")
)
df_angles$data_type <- factor(df_angles$data_type,
                               levels = c("Pure rotation", "Random"))

df_peak <- data.frame(
  peak      = c(r_rot$peak, r_rnd$peak),
  data_type = factor(c("Pure rotation", "Random"),
                     levels = c("Pure rotation", "Random"))
)

fig2 <- ggplot(df_angles, aes(x = angle)) +
  geom_histogram(aes(y = after_stat(density), fill = data_type),
                 bins = 30, alpha = 0.7, colour = "white") +
  geom_vline(xintercept = pi / 2, colour = "red",
             linetype = "dashed", linewidth = 0.9) +
  geom_vline(data = df_peak, aes(xintercept = peak, colour = data_type),
             linewidth = 1, show.legend = FALSE) +
  annotate("text", x = pi / 2 + 0.15, y = Inf,
           label = "\u03c0/2", vjust = 1.5, colour = "red", size = 3.5) +
  facet_wrap(~data_type, ncol = 2) +
  scale_x_continuous(
    breaks = c(-pi, -pi/2, 0, pi/2, pi),
    labels = c("-\u03c0", "-\u03c0/2", "0", "\u03c0/2", "\u03c0")
  ) +
  labs(
    title    = "Figure 2 \u2014 Angle distributions (\u03b8 between x and \u1e8b in jPC plane)",
    subtitle = "\u03b8 = \u03c0/2 \u2192 pure rotation  |  \u03b8 \u2248 0 \u2192 expansion",
    x = "Angle \u03b8 (rad)", y = "Density", fill = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "none")

ggsave(file.path(out_dir, "fig2_angle_distributions.png"),
       fig2, width = 9, height = 4, dpi = 120)

# ==============================================================================
# Figure 3: R2 comparison bar chart
# ==============================================================================

df_r2 <- data.frame(
  metric    = factor(rep(c("R\u00b2_unrestr", "R\u00b2_skew"), 2),
                     levels = c("R\u00b2_unrestr", "R\u00b2_skew")),
  value     = c(m_rot$R2_unrestr, m_rot$R2_skew,
                m_rnd$R2_unrestr, m_rnd$R2_skew),
  data_type = factor(rep(c("Pure rotation", "Random"), each = 2),
                     levels = c("Pure rotation", "Random"))
)

fig3 <- ggplot(df_r2, aes(x = metric, y = value, fill = data_type)) +
  geom_col(position = position_dodge(0.65), width = 0.55, alpha = 0.85) +
  geom_text(aes(label = sprintf("%.3f", value)),
            position = position_dodge(0.65), vjust = -0.4, size = 3.5) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50",
             linewidth = 0.6) +
  scale_y_continuous(limits = c(0, 1.12), expand = c(0, 0)) +
  labs(
    title = "Figure 3 \u2014 Fit quality: unrestricted vs skew-symmetric M",
    x = NULL, y = "R\u00b2", fill = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(file.path(out_dir, "fig3_R2_comparison.png"),
       fig3, width = 6, height = 4.5, dpi = 120)

# ==============================================================================
# Markdown summary
# ==============================================================================

md <- c(
  "# jPCA Test Visualizations",
  "",
  sprintf("_Generated: %s_", format(Sys.time(), "%Y-%m-%d %H:%M")),
  "",
  "## Surrogate data",
  "- **Rotation**: 2D circular orbit embedded in N=6 dims, \u03c9=0.2, noise_sd=0.05, C=4 conditions",
  "- **Random**: i.i.d. Gaussian, same shape",
  "",
  "---",
  "",
  "## Figure 1 \u2014 jPCA plane trajectories",
  "![trajectories](fig1_trajectories.png)",
  "Filled circles = t=0 (start). Pure rotation shows organized spirals; random shows noisy wandering.",
  "",
  "---",
  "",
  "## Figure 2 \u2014 Angle distributions",
  "![angles](fig2_angle_distributions.png)",
  "\u03b8 = angle between x(t) and \u1e8b(t) in the jPC plane. \u03b8 = \u03c0/2 \u2192 pure rotation; \u03b8 \u2248 0 \u2192 expansion.",
  "",
  "---",
  "",
  "## Figure 3 \u2014 R\u00b2 fit quality",
  "![R2](fig3_R2_comparison.png)",
  "R\u00b2_unrestr = unconstrained M; R\u00b2_skew = skew-symmetric constraint.",
  "",
  "---",
  "",
  "## Numerical summary",
  "",
  "| Metric | Rotation | Random |",
  "|--------|----------|--------|",
  sprintf("| R\u00b2_unrestr       | %.4f | %.4f |", m_rot$R2_unrestr, m_rnd$R2_unrestr),
  sprintf("| R\u00b2_skew         | %.4f | %.4f |", m_rot$R2_skew,    m_rnd$R2_skew),
  sprintf("| R\u00b2_ratio        | %.4f | %.4f |", r_rot$R2_ratio,   r_rnd$R2_ratio),
  sprintf("| peak angle (rad) | %.3f  | %.3f  |", r_rot$peak, r_rnd$peak),
  sprintf("| peak / (\u03c0/2)     | %.3f  | %.3f  |",
          r_rot$peak / (pi/2), r_rnd$peak / (pi/2)),
  "",
  "---",
  "",
  "## Algebraic test results",
  "",
  "| Test | Value |",
  "|------|-------|",
  sprintf("| M_skew + t(M_skew) max abs | %.2e |",
          max(abs(m_rot$M_skew + t(m_rot$M_skew)))),
  sprintf("| Re(eigenvalues) max abs    | %.2e |",
          max(abs(Re(eigen(m_rot$M_skew)$values)))),
  sprintf("| jPC1 \u00b7 jPC2               | %.2e |",
          sum(m_rot$W[1,] * m_rot$W[2,])),
  sprintf("| \u2016jPC1\u2016                    | %.10f |", sqrt(sum(m_rot$W[1,]^2))),
  sprintf("| \u2016jPC2\u2016                    | %.10f |", sqrt(sum(m_rot$W[2,]^2))),
  sprintf("| R\u00b2_skew \u2264 R\u00b2_unrestr     | %.4f \u2264 %.4f |",
          m_rot$R2_skew, m_rot$R2_unrestr)
)

cat(paste(md, collapse = "\n"),
    file = file.path(out_dir, "jpca_test_summary.md"))

cat("Done. Output written to visualizations/\n")
cat("  fig1_trajectories.png\n")
cat("  fig2_angle_distributions.png\n")
cat("  fig3_R2_comparison.png\n")
cat("  jpca_test_summary.md\n")
