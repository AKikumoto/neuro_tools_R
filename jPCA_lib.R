# jPCA_lib.R
# ==============================================================================
# R implementation of Churchland et al. (2012) jPCA
# MATLAB original: Shenoy Lab, Stanford
#
# Functions:
#   jpca_fit               -- fit jPCA: find rotation planes in population dynamics
#   jpca_transform         -- project data onto jPCA planes
#   jpca_rotation_strength -- angle distribution and R2_ratio for rotation strength
#
# Data convention:
#   X_list : list of matrices [N x T], one per condition
#   N = neurons/channels, T = time-points per condition
#   Minimum 3 conditions required (Churchland et al. 2012 §Methods).
#
# Reference:
#   Churchland MM, Cunningham JP, Kaufman MT, Foster JD, Nuyujukian P, Ryu SI,
#   Shenoy KV (2012). Neural population dynamics during reaching.
#   Nature 487, 51-56. DOI: 10.1038/nature11129
# ==============================================================================


# ==============================================================================
# Dependencies
# ==============================================================================

## %=% parsing list output to multiple variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %-%, extendToMatch(), g()
# EX) g(a, b, c)  %=%  list("hello", 123, list("apples, oranges"))

# Generic form
'%=%' = function(l, r, ...) UseMethod('%=%')

# Binary Operator
'%=%.lbunch' = function(l, r, ...) {
  Envir = as.environment(-1)

  if (length(r) > length(l))
    warning("RHS has more args than LHS. Only first", length(l), "used.")

  if (length(l) > length(r))  {
    #warning("LHS has more args than RHS. RHS will be repeated.")
    r <- extendToMatch(r, l) # repeating
  }

  for (II in 1:length(l)) {
    do.call('<-', list(l[[II]], r[[II]]), envir=Envir)
  }
}

# Used if LHS is larger than RHS
extendToMatch <- function(source, destin) {
  s <- length(source)
  d <- length(destin)

  # Assume that destin is a length when it is a single number and source is not
  if(d==1 && s>1 && !is.null(as.numeric(destin)))
    d <- destin

  dif <- d - s
  if (dif > 0) {
    source <- rep(source, ceiling(d/s))[1:d]
  }
  return (source)
}

# Grouping the left hand side
g = function(...) {
  List = as.list(substitute(list(...)))[-1L]
  class(List) = 'lbunch'
  return(List)
}


# ==============================================================================
# Functions
# ==============================================================================

# =========================================================================
# jpca_fit: FIT jPCA MODEL -- ROTATION PLANES IN POPULATION DYNAMICS
# =========================================================================
# [Role]:
#   Identify the low-dimensional subspace in which population activity
#   rotates most strongly. The central insight of jPCA is that preparatory
#   and movement dynamics are captured by a skew-symmetric dynamics matrix
#   M_skew, which has purely imaginary eigenvalues -- it only rotates,
#   never expands or contracts.
#
#   Churchland et al. (2012) §Methods: "We used jPCA to find a 6D subspace
#   (three jPCA planes) in which the population response rotated maximally."
#
# [Inputs]:
#   X_list   : list of matrices [N x T], one per condition.
#              N = neurons, T = time-points. Same N and T across conditions.
#   n_pcs    : integer, PCs to retain before fitting M_skew (default 6).
#              Must be even; each rotation plane uses one conjugate eigenpair.
#   normalize: logical, divide each PC by its std so XX^T ≈ s*I (default TRUE).
#              Required for skew-symmetrization to be the exact constrained
#              optimum; see ARCHITECTURE_demixed_j_PCA.md Sec. 15.
#
# [Outputs]:
#   $W          : matrix [2 x n_pcs], jPC axes for the dominant rotation plane.
#                 Rows are jPC1 and jPC2 (unit vectors in PC space).
#   $W_all      : matrix [n_pcs x n_pcs], jPC axes for all rotation planes.
#                 Rows 2k-1 and 2k form the k-th plane (k = 1..n_pcs/2).
#   $M_skew     : matrix [n_pcs x n_pcs], fitted skew-symmetric dynamics matrix.
#   $M_unrestr  : matrix [n_pcs x n_pcs], unconstrained OLS dynamics matrix.
#   $R2_skew    : scalar, R² of M_skew prediction of dX/dt.
#   $R2_unrestr : scalar, R² of unconstrained M prediction of dX/dt.
#   $pca        : prcomp result (rotation matrix for projecting new data).
#   $eig_freq   : vector [n_pcs], |Im(eigenvalues)| of M_skew, sorted descending.
#                 First entry = dominant rotation frequency.
#   $pc_std     : vector [n_pcs], per-PC standard deviations used for normalization.
#   $normalize, $n_pcs, $C, $T : metadata.
#
# [Algorithm]:
#   1. Stack conditions: X_full = [X_1 | ... | X_C]             [N x C*T]
#   2. Subtract cross-condition mean per time-point (removes shared evoked response).
#   3. PCA prefilter: project onto top n_pcs PCs                 [n_pcs x C*T]
#   4. Soft normalization: divide each PC row by its std (XX^T ≈ s*I).
#   5. Finite differences: dX[t] = X[t+1] - X[t], avoiding cross-condition seams.
#   6. Fit unconstrained dynamics: M_hat = dX X^T (X X^T)^{-1}  (OLS)
#   7. Skew-symmetrize: M_skew = (M_hat - M_hat^T) / 2          (Supp. eq. 3)
#   8. Eigendecompose M_skew (eigenvalues are purely imaginary ±iω).
#   9. Recover real rotation plane from first conjugate eigenpair:
#        jPC1 =  Re(v1) / ||Re(v1)||,   jPC2 = -Im(v1) / ||Im(v1)||
#      Orient jPC2 so mean rotation is counter-clockwise.
#
# [Notes]:
#   - Soft normalization (Step 4) is from the original MATLAB code, not stated
#     in the paper. Without it, skew-symmetrization is not the constrained
#     optimum when PC variances are unequal.
#   - Counter-clockwise orientation is conventional (Churchland MATLAB code).
#
# [Reference]:
#   Churchland et al. (2012) Nature 487, 51-56. Supplementary Methods.
# =========================================================================
jpca_fit <- function(X_list, n_pcs = 6, normalize = TRUE) {
  if (!is.list(X_list)) stop("X_list must be a list of matrices (one per condition)")

  N <- nrow(X_list[[1]])
  T <- ncol(X_list[[1]])
  C <- length(X_list)

  # Step 1: Stack all conditions -> [N x C*T]
  # Churchland et al. 2012, Supplementary Methods eq. 1
  X_full <- do.call(cbind, X_list)  # [N x C*T]

  # Step 2: Subtract cross-condition mean per timepoint
  # Removes variance shared across all conditions (evoked response)
  # mean over conditions at each timepoint: [N x T]
  X_cond_mean <- Reduce("+", X_list) / C
  X_cond_mean_rep <- do.call(cbind, rep(list(X_cond_mean), C))  # [N x C*T]
  X_centered <- X_full - X_cond_mean_rep  # [N x C*T]

  # Step 3: PCA prefilter -> keep top n_pcs PCs
  # prcomp expects observations in rows -> transpose
  pca <- prcomp(t(X_centered), center = FALSE)
  X_red <- t(pca$rotation[, 1:n_pcs]) %*% X_centered  # [n_pcs x C*T]

  # Soft normalization: equalize variance across PCs so XX^T ≈ s*I
  # This ensures skew-symmetrization of M_unrestr is the constrained optimum
  # (Churchland et al. 2012, MATLAB code; ARCHITECTURE Sec.15 open question)
  pc_std <- apply(X_red, 1, sd)
  pc_std[pc_std == 0] <- 1  # guard against zero-variance PCs
  if (normalize) {
    X_red <- sweep(X_red, 1, pc_std, "/")
  }

  # Step 4: Finite difference approximation of dX/dt
  # dX[t] ≈ X[t+1] - X[t]   (Churchland 2012, Supp. Methods)
  # Keep paired columns (drop last timepoint of each condition to avoid
  # bleeding across condition boundaries)
  keep_prev <- rep(TRUE, C * T)
  keep_next <- rep(TRUE, C * T)
  for (c in 1:C) {
    keep_prev[(c - 1) * T + T] <- FALSE  # last timepoint of condition c
    keep_next[(c - 1) * T + 1] <- FALSE  # first timepoint of condition c
  }
  X_prev <- X_red[, keep_prev, drop = FALSE]  # [n_pcs x C*(T-1)]
  X_next <- X_red[, keep_next, drop = FALSE]  # [n_pcs x C*(T-1)]
  dX     <- X_next - X_prev                   # dX/dt approximation

  # Step 5: Fit unconstrained M: dX ≈ M X
  # M_hat = dX X^T (X X^T)^{-1}   (ordinary least squares)
  XtX      <- X_prev %*% t(X_prev)              # [n_pcs x n_pcs]
  M_unrestr <- dX %*% t(X_prev) %*% solve(XtX)  # [n_pcs x n_pcs]

  # Step 6: Skew-symmetrize
  # M_skew = (M_hat - M_hat^T) / 2   (Churchland 2012, Supp. eq. 3)
  M_skew <- (M_unrestr - t(M_unrestr)) / 2

  # Step 7: R² of each fit
  dX_pred_unrestr <- M_unrestr %*% X_prev
  dX_pred_skew    <- M_skew    %*% X_prev
  SS_tot          <- sum(dX^2)
  R2_unrestr      <- 1 - sum((dX - dX_pred_unrestr)^2) / SS_tot
  R2_skew         <- 1 - sum((dX - dX_pred_skew)^2)    / SS_tot

  # Step 8: Eigendecomposition of M_skew
  # Eigenvalues are purely imaginary ±iω for skew-symmetric M
  eig       <- eigen(M_skew)
  eig_vals  <- eig$values   # complex, purely imaginary
  eig_vecs  <- eig$vectors  # complex

  # Sort by |Im(eigenvalue)| descending -> strongest rotation first
  eig_freq <- abs(Im(eig_vals))
  ord      <- order(eig_freq, decreasing = TRUE)
  eig_vals <- eig_vals[ord]
  eig_vecs <- eig_vecs[, ord, drop = FALSE]
  eig_freq <- eig_freq[ord]

  # Step 9: Recover real rotation plane from first complex conjugate pair
  # jPC1 =  Re(v1) / ||Re(v1)||,   jPC2 = -Im(v1) / ||Im(v1)||
  # (Churchland 2012, Supp. Methods; normalize to unit length)
  v1   <- eig_vecs[, 1]
  jPC1 <- Re(v1)
  jPC2 <- -Im(v1)
  jPC1 <- jPC1 / sqrt(sum(jPC1^2))
  jPC2 <- jPC2 / sqrt(sum(jPC2^2))

  # Orient so mean rotation is counter-clockwise:
  # compute mean cross product x ∧ ẋ in the jPC1/jPC2 plane
  x1  <- t(jPC1) %*% X_prev   # [1 x C*(T-1)]
  x2  <- t(jPC2) %*% X_prev
  dx1 <- t(jPC1) %*% dX
  dx2 <- t(jPC2) %*% dX
  mean_cross <- mean(x1 * dx2 - x2 * dx1)
  if (mean_cross < 0) {
    jPC2 <- -jPC2  # flip to make counter-clockwise
  }

  W <- rbind(jPC1, jPC2)  # [2 x n_pcs]

  # Build W_all: real rotation planes from all conjugate pairs
  # Each pair (v, v̄) at indices 2k-1, 2k gives one rotation plane
  n_pairs <- floor(n_pcs / 2)
  W_all   <- matrix(0, nrow = n_pcs, ncol = n_pcs)
  for (k in 1:n_pairs) {
    vk  <- eig_vecs[, 2 * k - 1]
    row1 <-  Re(vk) / sqrt(sum(Re(vk)^2))
    row2 <- -Im(vk) / sqrt(sum(Im(vk)^2))
    W_all[2 * k - 1, ] <- row1
    W_all[2 * k,     ] <- row2
  }

  return(list(
    W          = W,
    W_all      = W_all,
    M_skew     = M_skew,
    M_unrestr  = M_unrestr,
    R2_skew    = R2_skew,
    R2_unrestr = R2_unrestr,
    pca        = pca,
    pc_std     = pc_std,
    normalize  = normalize,
    eig_freq   = eig_freq,
    n_pcs      = n_pcs,
    C          = C,
    T          = T
  ))
}


# =========================================================================
# jpca_transform: PROJECT DATA ONTO jPCA PLANES
# =========================================================================
# [Role]:
#   Project new or training data onto the jPC rotation plane identified by
#   jpca_fit. Applies the same preprocessing steps (centering, PCA, soft
#   normalization) as jpca_fit so that projections live in the same
#   coordinate space as the fitted rotation plane.
#
# [Inputs]:
#   X_list : list of matrices [N x T], one per condition.
#            Can be training data (same as jpca_fit) or held-out test data.
#   model  : list output of jpca_fit.
#
# [Outputs]:
#   $proj      : matrix [2 x C*T], all conditions stacked, projected onto
#                the first jPCA plane (row 1 = jPC1, row 2 = jPC2).
#   $proj_list : list of C matrices [2 x T], one per condition.
#
# [Algorithm]:
#   1. Stack and subtract cross-condition mean (same as jpca_fit Steps 1-2).
#   2. Project onto stored PCA rotation: X_red = pca$rotation[,1:n_pcs]^T X_centered.
#   3. Apply same soft normalization (divide by pc_std from fit).
#   4. Project onto jPCA plane: proj = W X_red     [2 x C*T]
#   5. Split into per-condition list.
#
# [Notes]:
#   - Uses the STORED pca$rotation from jpca_fit, not re-fitted PCA.
#     Test data is thus projected into the training PCA space.
#   - When X_list is the same data used in jpca_fit, $proj reproduces the
#     training trajectories exactly.
#
# [Reference]:
#   Churchland et al. (2012) Supplementary Methods
# =========================================================================
jpca_transform <- function(X_list, model) {
  C <- length(X_list)
  T <- ncol(X_list[[1]])

  # Re-apply same centering as in jpca_fit
  X_full      <- do.call(cbind, X_list)
  X_cond_mean <- Reduce("+", X_list) / C
  X_cond_mean_rep <- do.call(cbind, rep(list(X_cond_mean), C))
  X_centered  <- X_full - X_cond_mean_rep  # [N x C*T]

  # Project onto PCA space (using stored rotation matrix)
  X_red <- t(model$pca$rotation[, 1:model$n_pcs]) %*% X_centered  # [n_pcs x C*T]

  # Apply same soft normalization as in jpca_fit
  if (model$normalize) {
    X_red <- sweep(X_red, 1, model$pc_std, "/")
  }

  # Project onto jPC plane: W [2 x n_pcs] %*% X_red [n_pcs x C*T] -> [2 x C*T]
  proj <- model$W %*% X_red

  # Split back into per-condition list
  proj_list <- vector("list", C)
  for (c in 1:C) {
    idx <- ((c - 1) * T + 1):(c * T)
    proj_list[[c]] <- proj[, idx, drop = FALSE]  # [2 x T]
  }
  if (!is.null(names(X_list))) names(proj_list) <- names(X_list)

  return(list(
    proj      = proj,
    proj_list = proj_list
  ))
}


# =========================================================================
# jpca_rotation_strength: ANGLE DISTRIBUTION AND R2_RATIO FOR ROTATION
# =========================================================================
# [Role]:
#   Quantify how strongly population dynamics rotate in the jPCA plane.
#   For each time point and condition, compute the angle theta between the
#   position vector x and the velocity vector dx/dt in the 2D jPC plane.
#   Pure rotation gives theta near pi/2; pure expansion gives theta near 0.
#
#   Also returns R2_ratio = R2_skew / R2_unrestr, the fraction of explainable
#   dynamics variance captured by rotational (skew-symmetric) dynamics alone.
#   R2_ratio = 1 means dynamics are purely rotational.
#
#   Churchland et al. (2012) Fig. 6a shows this distribution peaked near pi/2
#   for M1 and PMd data across multiple monkeys and sessions.
#
# [Inputs]:
#   proj_result : list output of jpca_transform (must contain $proj_list).
#   model       : list output of jpca_fit (must contain $R2_skew, $R2_unrestr).
#
# [Outputs]:
#   $angles  : numeric vector, theta in radians for all (time x condition) pairs.
#              theta in (-pi, pi); distribution peaked near pi/2 = strong rotation.
#   $peak    : scalar, mode of the angle distribution (histogram with 30 bins).
#   $R2_ratio: scalar, R2_skew / R2_unrestr.
#
# [Algorithm]:
#   For each condition c, for t in 1:(T-1):
#     x  = proj_list[[c]][, t]           position at t          [2]
#     dx = proj_list[[c]][, t+1] - x     finite-difference velocity [2]
#     cross = x[1]*dx[2] - x[2]*dx[1]   2D cross product (signed area)
#     dot   = x[1]*dx[1] + x[2]*dx[2]   inner product
#     theta = atan2(cross, dot)
#   Peak = histogram mode (30 bins over angles).
#
# [Notes]:
#   - theta = pi/2: x and dx are orthogonal (pure rotation).
#   - theta = 0:    x and dx are parallel (pure expansion/contraction).
#   - The 2D cross product is the z-component of the 3D cross product,
#     giving the signed area (positive = counter-clockwise).
#
# [Reference]:
#   Churchland et al. (2012) Nature 487, 51-56. Fig. 6a.
# =========================================================================
jpca_rotation_strength <- function(proj_result, model) {
  # Churchland et al. 2012, Fig. 6a
  # theta(t,c) = atan2(x ∧ ẋ, x · ẋ)
  # where ∧ is the 2D cross product and · is dot product

  proj_list <- proj_result$proj_list
  C <- length(proj_list)
  T <- ncol(proj_list[[1]])

  angles <- c()
  for (cond in 1:C) {
    X_cond <- proj_list[[cond]]  # [2 x T]
    # finite differences (exclude last timepoint)
    x  <- X_cond[, 1:(T - 1), drop = FALSE]   # [2 x T-1]
    dx <- X_cond[, 2:T, drop = FALSE] - x      # [2 x T-1]

    # 2D cross product: x[1]*dx[2] - x[2]*dx[1]  (signed area)
    cross <- x[1, ] * dx[2, ] - x[2, ] * dx[1, ]
    dot   <- x[1, ] * dx[1, ] + x[2, ] * dx[2, ]

    angles <- c(angles, atan2(cross, dot))
  }

  # Mode via histogram
  h    <- hist(angles, breaks = 30, plot = FALSE)
  peak <- h$mids[which.max(h$counts)]

  R2_ratio <- model$R2_skew / model$R2_unrestr

  return(list(
    angles   = angles,
    peak     = peak,
    R2_ratio = R2_ratio
  ))
}
