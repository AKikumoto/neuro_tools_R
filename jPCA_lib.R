# jPCA library
# ------------------------------------------------------------------------------
# Churchland et al. (2012) Nature 487, 51–56
# [Done]
# - 
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

jpca_fit <- function(X_list, n_pcs = 6, normalize = TRUE) {
  # [model] = jpca_fit(X_list, n_pcs)
  # Fit jPCA model: find rotation planes in population dynamics
  #
  # model$W         : jPC axes [2 × n_pcs] for first rotation plane
  # model$W_all     : all jPC pairs [n_pcs × n_pcs]
  # model$M_skew    : skew-symmetric dynamics matrix [n_pcs × n_pcs]
  # model$M_unrestr : unconstrained dynamics matrix [n_pcs × n_pcs]
  # model$R2_skew   : R² of M_skew fit
  # model$R2_unrestr: R² of unconstrained M fit
  # model$pca       : prcomp result (for projecting new data)
  # model$eig_freq  : rotation frequencies |Im(eigenvalues)|, sorted descending
  #
  # X_list: list of matrices [N × T], one per condition
  # n_pcs:  number of PCs to retain before fitting M_skew
  # --------------------------------------------------------
  if (!is.list(X_list)) stop("X_list must be a list of matrices (one per condition)")

  N <- nrow(X_list[[1]])
  T <- ncol(X_list[[1]])
  C <- length(X_list)

  # Step 1: Stack all conditions → [N × C*T]
  # Churchland et al. 2012, Supplementary Methods eq. 1
  X_full <- do.call(cbind, X_list)  # [N × C*T]

  # Step 2: Subtract cross-condition mean per timepoint
  # Removes variance shared across all conditions (evoked response)
  # mean over conditions at each timepoint: [N × T]
  X_cond_mean <- Reduce("+", X_list) / C
  X_cond_mean_rep <- do.call(cbind, rep(list(X_cond_mean), C))  # [N × C*T]
  X_centered <- X_full - X_cond_mean_rep  # [N × C*T]

  # Step 3: PCA prefilter → keep top n_pcs PCs
  # prcomp expects observations in rows → transpose
  pca <- prcomp(t(X_centered), center = FALSE)
  X_red <- t(pca$rotation[, 1:n_pcs]) %*% X_centered  # [n_pcs × C*T]

  # Soft normalization: equalize variance across PCs so XX^T ≈ s*I
  # This ensures skew-symmetrization of M_unrestr is the constrained optimum
  # (Churchland et al. 2012, MATLAB code; ARCHITECTURE Sec.15 open question)
  pc_std <- apply(X_red, 1, sd)
  pc_std[pc_std == 0] <- 1  # guard against zero-variance PCs
  if (normalize) {
    X_red <- sweep(X_red, 1, pc_std, "/")
  }

  # Step 4: Finite difference approximation of Ẋ
  # Ẋ[t] ≈ X[t+1] - X[t]   (Churchland 2012, Supp. Methods)
  # Keep paired columns (drop last timepoint of each condition to avoid
  # bleeding across condition boundaries)
  keep_prev <- rep(TRUE, C * T)
  keep_next <- rep(TRUE, C * T)
  for (c in 1:C) {
    keep_prev[(c - 1) * T + T] <- FALSE  # last timepoint of condition c
    keep_next[(c - 1) * T + 1] <- FALSE  # first timepoint of condition c
  }
  X_prev <- X_red[, keep_prev, drop = FALSE]  # [n_pcs × C*(T-1)]
  X_next <- X_red[, keep_next, drop = FALSE]  # [n_pcs × C*(T-1)]
  dX     <- X_next - X_prev                   # Ẋ approximation

  # Step 5: Fit unconstrained M: Ẋ ≈ M X
  # M_hat = Ẋ Xᵀ (X Xᵀ)⁻¹   (ordinary least squares)
  XtX      <- X_prev %*% t(X_prev)              # [n_pcs × n_pcs]
  M_unrestr <- dX %*% t(X_prev) %*% solve(XtX)  # [n_pcs × n_pcs]

  # Step 6: Skew-symmetrize
  # M_skew = (M_hat - M_hatᵀ) / 2   (Churchland 2012, Supp. eq. 3)
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

  # Sort by |Im(eigenvalue)| descending → strongest rotation first
  eig_freq <- abs(Im(eig_vals))
  ord      <- order(eig_freq, decreasing = TRUE)
  eig_vals <- eig_vals[ord]
  eig_vecs <- eig_vecs[, ord, drop = FALSE]
  eig_freq <- eig_freq[ord]

  # Step 9: Recover real rotation plane from first complex conjugate pair
  # jPC1 =  2 Re(v₁)   jPC2 = -2 Im(v₁)
  # (Churchland 2012, Supp. Methods; normalize to unit length)
  v1   <- eig_vecs[, 1]
  jPC1 <- Re(v1)
  jPC2 <- -Im(v1)
  jPC1 <- jPC1 / sqrt(sum(jPC1^2))
  jPC2 <- jPC2 / sqrt(sum(jPC2^2))

  # Orient so mean rotation is counter-clockwise:
  # compute mean cross product x ∧ ẋ in the jPC1/jPC2 plane
  x1  <- t(jPC1) %*% X_prev   # [1 × C*(T-1)]
  x2  <- t(jPC2) %*% X_prev
  dx1 <- t(jPC1) %*% dX
  dx2 <- t(jPC2) %*% dX
  mean_cross <- mean(x1 * dx2 - x2 * dx1)
  if (mean_cross < 0) {
    jPC2 <- -jPC2  # flip to make counter-clockwise
  }

  W <- rbind(jPC1, jPC2)  # [2 × n_pcs]

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


jpca_transform <- function(X_list, model) {
  # [result] = jpca_transform(X_list, model)
  # Project data onto jPCA planes
  #
  # result$proj      : matrix [2 × C*T], projection onto first jPCA plane
  # result$proj_list : list of [2 × T] matrices, one per condition
  #
  # X_list: list of condition matrices [N × T] (same format as jpca_fit input)
  # model:  output of jpca_fit
  # --------------------------------------------------------
  C <- length(X_list)
  T <- ncol(X_list[[1]])

  # Re-apply same centering as in jpca_fit
  X_full      <- do.call(cbind, X_list)
  X_cond_mean <- Reduce("+", X_list) / C
  X_cond_mean_rep <- do.call(cbind, rep(list(X_cond_mean), C))
  X_centered  <- X_full - X_cond_mean_rep  # [N × C*T]

  # Project onto PCA space (using stored rotation matrix)
  X_red <- t(model$pca$rotation[, 1:model$n_pcs]) %*% X_centered  # [n_pcs × C*T]

  # Apply same soft normalization as in jpca_fit
  if (model$normalize) {
    X_red <- sweep(X_red, 1, model$pc_std, "/")
  }

  # Project onto jPC plane: W [2 × n_pcs] %*% X_red [n_pcs × C*T] → [2 × C*T]
  proj <- model$W %*% X_red

  # Split back into per-condition list
  proj_list <- vector("list", C)
  for (c in 1:C) {
    idx <- ((c - 1) * T + 1):(c * T)
    proj_list[[c]] <- proj[, idx, drop = FALSE]  # [2 × T]
  }
  if (!is.null(names(X_list))) names(proj_list) <- names(X_list)

  return(list(
    proj      = proj,
    proj_list = proj_list
  ))
}


jpca_rotation_strength <- function(proj_result, model) {
  # [result] = jpca_rotation_strength(proj_result, model)
  # Compute rotation strength: angle distribution between x and ẋ in jPCA plane
  #
  # result$angles  : vector of angles θ (radians), one per time×condition
  #                  peak near π/2 = pure rotation, near 0 = expansion
  # result$peak    : mode of angle distribution (histogram peak)
  # result$R2_ratio: R²_skew / R²_unrestr  (1.0 = dynamics fully rotational)
  #
  # proj_result: output of jpca_transform
  # model:       output of jpca_fit
  # --------------------------------------------------------
  # Churchland et al. 2012, Fig. 6a
  # θ(t,c) = atan2(x ∧ ẋ, x · ẋ)
  # where ∧ is the 2D cross product and · is dot product

  proj_list <- proj_result$proj_list
  C <- length(proj_list)
  T <- ncol(proj_list[[1]])

  angles <- c()
  for (cond in 1:C) {
    X_cond <- proj_list[[cond]]  # [2 × T]
    # finite differences (exclude last timepoint)
    x  <- X_cond[, 1:(T - 1), drop = FALSE]   # [2 × T-1]
    dx <- X_cond[, 2:T, drop = FALSE] - x      # [2 × T-1]

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
