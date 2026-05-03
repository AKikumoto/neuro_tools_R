# demixed_PCA library
# ------------------------------------------------------------------------------
# R re-implementation of Kobak et al. (2016) demixed PCA
# Python original: https://github.com/machenslab/dPCA
#
# Core functions:
#   dpca_get_marginalizations  -- enumerate all parameter subsets
#   dpca_marginalize           -- decompose X into orthogonal marginals
#   dpca_fit                   -- solve for encoder/decoder pairs (P, D)
#   dpca_transform             -- project data to latent components
#   dpca_inverse_transform     -- reconstruct data from latent components
#   dpca_reconstruct           -- fit_transform + inverse in one call
#   dpca_significance          -- cross-validated significance masks
#   dpca_plot                  -- ggplot2 panel: each marginalization × component
#
# Data convention:
#   X  : array [N, d1, d2, ...] where N = neurons/channels
#        and d1, d2, ... are parameter axes (e.g. S stimuli, T time-points)
#   labels : character string, one char per parameter axis (e.g. "st")
#
# Reference:
#   Kobak D, Brendel W, et al. (2016). Demixed principal component analysis
#   of neural population data. eLife 5:e10989.
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
prep_cond_avg <- function(d, blIDX, grpIDX) {
  # [XnL] = prep_cond_avg(d, blIDX, grpIDX)
  # Condition-averaged data per balance fold → list of [N x T] matrices
  #
  # XnL : list of matrices [N x T], one per balance fold
  #
  # d      : array [n_trial x n_time x N]
  # blIDX  : balance fold ID per trial (ds_bl[["balanceID"]])
  # grpIDX : condition label per trial (ds_bl[[modelV]])
  # --------------------------------------------------------
  Xn  <- narray::map(d, along=1, mean,
                     subsets=paste0(blIDX,"_",grpIDX))
  Xn  <- Xn[order(rownames(Xn)),,]
  Xn  <- narray::split(Xn, along=1,
                       subsets=rep(unique(blIDX), each=dim(Xn)[1]/2))
  XnL <- lapply(Xn, function(x){t(dimAdj(x,c(1,2)))})
  return(XnL)
}


# ==============================================================================
# dPCA core functions
# ==============================================================================

# ------------------------------------------------------------------------------
# dpca_get_marginalizations
# ------------------------------------------------------------------------------
# Returns all non-empty subsets of the parameter axes as a named list.
# Each element is a sorted integer vector of 0-based parameter axis indices.
#
# Parameters:
#   labels : character string, e.g. "st"
#
# Returns:
#   Named list, e.g. list(s=0L, t=1L, st=c(0L,1L))
#   The indices refer to the PARAMETER axes (i.e. dim 2,3,... of X).
#
# Example (labels="st"):
#   list(s=0, t=1, st=c(0,1))
# ------------------------------------------------------------------------------
dpca_get_marginalizations <- function(labels) {
  K <- nchar(labels)
  chars <- strsplit(labels, "")[[1]]

  # generate all non-empty subsets (powerset minus empty set)
  out <- list()
  for (r in seq_len(K)) {
    combos <- combn(K, r, simplify = FALSE)
    for (combo in combos) {
      key <- paste(chars[combo], collapse = "")
      out[[key]] <- as.integer(combo - 1L)   # 0-based parameter axis indices
    }
  }
  out
}


# ------------------------------------------------------------------------------
# dpca_marginalize
# ------------------------------------------------------------------------------
# Decomposes the centered data array X into orthogonal marginalizations.
#
# ANOVA-style decomposition (e.g., labels="st", X=[N,S,T]):
#   X_s[n,s,t] = mean_t(Xc[n,s,:])           stimulus effect (constant over t)
#   X_t[n,s,t] = mean_s(Xc[n,:,t])           time effect    (constant over s)
#   X_st        = Xc - X_s - X_t              interaction
#   Sum: X_s + X_t + X_st = Xc  (orthogonal partition of centered variance)
#
# Algorithm follows Kobak et al. (2016) Appendix:
#   1. pre_mean[[key]] = mean of Xc over the axes IN key  
#      (shape with 1 at those axes, keeps all others)
#   2. base = pre_mean[[complement(phi)]] = mean over axes NOT in phi
#   3. Subtract all strict sub-marginals (inclusion-exclusion) to ensure
#      orthogonality
#   4. Expand singletons to full dim; flatten to [N, prod(dx)]
#
# Parameters:
#   X      : array [N, d1, d2, ...]  (neurons x parameters)
#   labels : character string, one char per parameter axis
#
# Returns:
#   Named list of matrices [N, prod(d1,...)] — one per marginalization.
# ------------------------------------------------------------------------------
dpca_marginalize <- function(X, labels) {
  N     <- dim(X)[1]
  dx    <- dim(X)[-1]
  K     <- length(dx)
  chars <- strsplit(labels, "")[[1]]
  dX    <- dim(X)            # c(N, d1, d2, ...)

  # center X: subtract per-neuron mean across all conditions
  Xflat <- matrix(X, nrow = N)
  A_res <- array(Xflat - rowMeans(Xflat), dim = dX)

  # helper: expand singleton dims to target_dim (R-safe broadcast)
  # Moves each singleton dim to last position, replicates, moves back.
  expand_to <- function(M, target_dim) {
    for (ax in which(dim(M) == 1L & target_dim > 1L)) {
      d     <- length(dim(M))
      n_rep <- target_dim[ax]
      perm  <- if (ax < d) c(seq_len(d)[-ax], ax) else seq_len(d)
      Mperm <- aperm(M, perm)
      Mperm <- array(rep(as.vector(Mperm), n_rep),
                     dim = c(dim(Mperm)[-d], n_rep))
      M     <- aperm(Mperm, order(perm))
    }
    M
  }

  margs <- dpca_get_marginalizations(labels)

  # pre_mean[[key]] = mean of A_res over parameter axes in 'key'
  # Shape = dX with 1 at each axis listed in key.
  pre_mean <- list()
  for (key in names(margs)) {
    phi       <- margs[[key]]
    ax_r_list <- phi + 2L          # R dim indices (N is dim 1; params start at 2)

    if (length(phi) == 1L) {
      ax_r <- ax_r_list
      keep <- setdiff(seq_len(K + 1L), ax_r)   # dims to keep (sorted)
      M    <- apply(A_res, keep, mean)           # shape = dX[keep]
      pre_mean[[key]] <- array(as.vector(M), dim = replace(dX, ax_r, 1L))

    } else {
      # accumulate: average parent's pre_mean over the last new axis
      parent_key <- substr(key, 1L, nchar(key) - 1L)
      last_ax_r  <- phi[length(phi)] + 2L
      par_arr    <- pre_mean[[parent_key]]
      keep       <- setdiff(seq_len(K + 1L), last_ax_r)
      M          <- apply(par_arr, keep, mean)
      pre_mean[[key]] <- array(as.vector(M), dim = replace(dim(par_arr), last_ax_r, 1L))
    }
  }

  # Inclusion-exclusion to get orthogonal marginals (all stored at full dX).
  Xmargs <- list()
  for (key in names(margs)) {
    key_without <- paste(setdiff(chars, strsplit(key, "")[[1L]]), collapse = "")
    base <- if (nchar(key_without) > 0L) {
      expand_to(pre_mean[[key_without]], dX)
    } else {
      A_res
    }

    if (length(margs[[key]]) > 1L) {
      subkeys <- unlist(lapply(
        seq_len(length(margs[[key]]) - 1L),
        function(r) apply(combn(strsplit(key, "")[[1L]], r), 2L, paste, collapse = "")
      ))
      out <- base
      for (sk in subkeys) out <- out - Xmargs[[sk]]
      Xmargs[[key]] <- out
    } else {
      Xmargs[[key]] <- base
    }
  }

  # flatten to [N, prod(dx)]
  lapply(Xmargs, function(M) matrix(M, nrow = N))
}


# ------------------------------------------------------------------------------
# dpca_fit
# ------------------------------------------------------------------------------
# Fit a dPCA model:  find encoder P and decoder D for each marginalization.
#
# The algorithm (Kobak et al. 2016, Appendix):
#   For each marginalization phi:
#     C_phi = X_phi_flat  %*%  pinv(X_flat)          [N × N]
#     Fact  = C_phi %*% X_flat                        [N × (prod dx)]
#     SVD:   Fact = U S V^T   (truncated to n_components)
#     P_phi = U                                       [N × k]  (encoder)
#     D_phi = (U^T %*% C_phi)^T  = C_phi^T %*% U    [N × k]  (decoder)
#
#   Transform:  Z_phi = t(D_phi) %*% X_flat           [k × (prod dx)]
#   Reconstruct: P_phi %*% Z_phi                      [N × (prod dx)]
#
# Note: P and D are distinct (unlike standard PCA). This is what "demixed" means:
#   D decodes a SINGLE parameter combination; P encodes back to full space.
#
# Parameters:
#   X           : array [N, d1, d2, ...]
#   labels      : character string
#   n_components: integer (same for all) or named list per marginalization
#   regularizer : numeric >= 0;  ridge penalty = regularizer * var(X)
#                 (set 0 for no regularization)
#
# Returns:  list with elements
#   P, D   : named lists of matrices [N × k], one per marginalization
#   labels : the labels string
#   dx     : parameter dimensions
#   N      : number of neurons
#   margs  : marginalization list from dpca_get_marginalizations
#   explained_variance_ratio : named list of numeric vectors (set after transform)
# ------------------------------------------------------------------------------
dpca_fit <- function(X, labels, n_components = 10, regularizer = 0) {
  N  <- dim(X)[1]
  dx <- dim(X)[-1]

  # center
  Xflat <- matrix(X, nrow = N)
  Xc    <- Xflat - rowMeans(Xflat)

  # marginalize
  mXs <- dpca_marginalize(X, labels)

  # add regularization: append lambda * I to X and zeros to each X_phi
  # this is the standard Tikhonov trick: argmin ||X_phi - D P^T X||^2 + lam||P||^2
  lam <- if (regularizer > 0) regularizer * sum(Xc^2) else 0

  if (lam > 0) {
    regXc  <- cbind(Xc, lam * diag(N))
    regmXs <- lapply(mXs, function(M) cbind(M, matrix(0, N, N)))
  } else {
    regXc  <- Xc
    regmXs <- mXs
  }

  # pseudoinverse of (regularized) X
  pinvX <- MASS::ginv(regXc)                 # [prod(dx) × N]  (or [(prod dx + N) × N])

  # compute encoder P and decoder D for each marginalization
  P <- list()
  D <- list()

  for (key in names(mXs)) {
    mX <- regmXs[[key]]                      # [N × (prod dx)]  or  [N × (prod dx + N)]

    # C_phi: how does the marginalization "project" linearly onto neurons?
    C  <- mX %*% pinvX                       # [N × N]

    # low-rank structure lives in C %*% X
    Fact <- C %*% regXc                      # [N × (prod dx)]

    # truncated SVD
    k  <- if (is.list(n_components)) n_components[[key]] else n_components
    k  <- min(k, N, ncol(Fact))

    sv <- svd(Fact, nu = k, nv = 0)

    U      <- sv$u[ , seq_len(k), drop = FALSE]   # [N × k]
    P[[key]] <- U
    D[[key]] <- t(t(U) %*% C)                     # [N × k]  =  C^T %*% U
  }

  structure(
    list(P = P, D = D, labels = labels, dx = dx, N = N,
         margs = dpca_get_marginalizations(labels),
         n_components = n_components, regularizer = regularizer),
    class = "dpca_model"
  )
}


# ------------------------------------------------------------------------------
# dpca_transform
# ------------------------------------------------------------------------------
# Project data X onto the dPCA components found during fit.
#
# Z_phi = t(D_phi) %*% X_flat     [k × (d1*d2*...)]
# then reshaped to [k, d1, d2, ...]
#
# Parameters:
#   X     : array [N, d1, d2, ...]   (same shape used in dpca_fit)
#   model : result of dpca_fit
#
# Returns:
#   Named list of arrays [k, d1, d2, ...], one per marginalization.
#   Also stores explained_variance_ratio in model environment (invisible side
#   effect — attach to the returned list as attribute "var_ratio").
# ------------------------------------------------------------------------------
dpca_transform <- function(X, model) {
  N    <- model$N
  dx   <- model$dx
  Xc   <- matrix(X, nrow = N)
  Xc   <- Xc - rowMeans(Xc)

  total_var <- sum(Xc^2)

  Z    <- list()
  evar <- list()

  for (key in names(model$D)) {
    Dk    <- model$D[[key]]                          # [N × k]
    Zflat <- t(Dk) %*% Xc                           # [k × prod(dx)]

    # explained variance per component
    evar[[key]] <- apply(Zflat, 1, function(z) sum(z^2) / total_var)

    # reshape to [k, d1, d2, ...]
    k <- ncol(Dk)
    Z[[key]] <- array(Zflat, dim = c(k, dx))
  }

  attr(Z, "explained_variance_ratio") <- evar
  Z
}


# ------------------------------------------------------------------------------
# dpca_inverse_transform
# ------------------------------------------------------------------------------
# Reconstruct the full [N, d1, d2, ...] data from latent components Z_phi.
#
# X_recon = P_phi %*% Z_phi_flat
#
# Parameters:
#   Z             : array [k, d1, d2, ...] (one element from dpca_transform output)
#   model         : result of dpca_fit
#   marginalization : character key, e.g. "s"
#
# Returns:  array [N, d1, d2, ...]
# ------------------------------------------------------------------------------
dpca_inverse_transform <- function(Z, model, marginalization) {
  Pk    <- model$P[[marginalization]]                 # [N × k]
  dx    <- model$dx
  k     <- ncol(Pk)
  Zflat <- matrix(Z, nrow = k)                       # [k × prod(dx)]

  Xrec  <- Pk %*% Zflat                              # [N × prod(dx)]
  array(Xrec, dim = c(model$N, dx))
}


# ------------------------------------------------------------------------------
# dpca_reconstruct
# ------------------------------------------------------------------------------
# Convenience: transform then inverse_transform for one marginalization.
#
# Returns: array [N, d1, d2, ...]
# ------------------------------------------------------------------------------
dpca_reconstruct <- function(X, model, marginalization) {
  Z <- dpca_transform(X, model)[[marginalization]]
  dpca_inverse_transform(Z, model, marginalization)
}


# ------------------------------------------------------------------------------
# dpca_significance
# ------------------------------------------------------------------------------
# Cross-validated significance analysis.
#
# For each marginalization phi (except the full-label one):
#   1. Split trials into train/test sets (n_splits times)
#   2. Fit dPCA on train, project test
#   3. Classify test labels using nearest-mean classifier in latent space
#   4. Compare classification accuracy against shuffled-label null distribution
#   5. Time-points where true_score > max_shuffle_score are significant
#
# Parameters:
#   X           : array [N, d1, ..., T]   trial-averaged
#   trialX      : array [n_trials, N, d1, ..., T]  trial-by-trial
#   model       : result of dpca_fit (or NULL to re-fit each time)
#   labels      : character string (used if model is NULL)
#   n_shuffles  : integer, number of label shuffles (p-value = 1/n_shuffles)
#   n_splits    : integer, train-test splits per evaluation
#   n_consecutive: require this many consecutive significant time-points
#   axis        : integer, axis index (1-based within Z) over which significance
#                 is evaluated; typically the time axis = last parameter axis
#   n_components: passed to dpca_fit if model is NULL
#
# Returns:
#   Named list of logical matrices [k × T], TRUE where significant.
# ------------------------------------------------------------------------------
dpca_significance <- function(X, trialX,
                               model       = NULL,
                               labels      = NULL,
                               n_shuffles  = 100,
                               n_splits    = 100,
                               n_consecutive = 1,
                               axis        = NULL,
                               n_components = 10) {

  if (is.null(model)) {
    if (is.null(labels)) stop("Provide either 'model' or 'labels'.")
    model <- dpca_fit(X, labels, n_components = n_components)
  }

  labels <- model$labels
  N      <- model$N
  dx     <- model$dx
  K      <- length(dx)

  # keys to test (exclude the time-only marginalization if desired, per original)
  # convention: skip the last single-label marginalization = "last label alone"
  # (Cannot classify over the axis you split along)
  keys <- names(model$margs)
  time_key <- substr(labels, nchar(labels), nchar(labels))   # last char
  keys <- setdiff(keys, time_key)

  # identify the "class" axis = first parameter axis by convention
  # (axis over which we classify, e.g., stimulus = axis 1 after N)
  class_ax <- 1L   # 0-based parameter axis index

  # helper: single train/test split from trialX
  train_test_split <- function(trialX) {
    # trialX: [n_trials, N, d1, ..., T]
    n_tr <- dim(trialX)[1]
    total_conditions <- prod(dim(trialX)[-c(1, 2, length(dim(trialX)))])

    dims    <- dim(trialX)[-1]   # [N, d1, ..., T]
    Xtest   <- array(0, dim = dims)
    Xmat    <- matrix(trialX, nrow = n_tr)   # [n_trials, N*d1*...*T]

    # random held-out trial index (same for all conditions; simplified)
    hold_idx <- sample.int(n_tr, 1L)

    # test: one held-out trial
    Xtest <- array(Xmat[hold_idx, ], dim = dims)

    # train: mean of the remaining trials
    remain <- setdiff(seq_len(n_tr), hold_idx)
    Xtrain <- array(colMeans(Xmat[remain, , drop = FALSE]), dim = dims)

    list(train = Xtrain, test = Xtest)
  }

  # helper: nearest-mean classifier over class_ax for last dim = time
  # class_means: [n_class, T],  test: [n_class, T]
  # returns accuracy vector [T]
  nearest_mean_acc <- function(class_means, test) {
    n_class <- nrow(class_means)
    T       <- ncol(class_means)
    acc     <- numeric(T)
    for (t in seq_len(T)) {
      correct <- 0
      for (p in seq_len(n_class)) {
        dists  <- abs(class_means[ , t] - test[p, t])
        pred   <- which.min(dists)
        if (pred == p) correct <- correct + 1
      }
      acc[t] <- correct / n_class
    }
    acc
  }

  # compute mean classification accuracy over n_splits
  compute_score <- function(X, trialX) {
    sc <- lapply(keys, function(k) {
      k_val <- model$n_components
      if (is.list(k_val)) k_val <- k_val[[k]]
      matrix(0, nrow = k_val, ncol = dx[K])    # [n_comp, T]
    })
    names(sc) <- keys

    for (split in seq_len(n_splits)) {
      sp      <- train_test_split(trialX)
      m_tmp   <- dpca_fit(sp$train, labels,
                          n_components = model$n_components,
                          regularizer  = model$regularizer)
      Z_train <- dpca_transform(sp$train, m_tmp)
      Z_valid <- dpca_transform(sp$test,  m_tmp)

      for (key in keys) {
        k_val     <- nrow(m_tmp$P[[key]])
        zt        <- Z_train[[key]]    # [k, d1, ..., T]
        zv        <- Z_valid[[key]]

        for (comp in seq_len(dim(zt)[1])) {
          # flatten class axis → [n_class, T]
          zt_c <- matrix(zt[comp, , ], nrow = dx[class_ax + 1L])
          zv_c <- matrix(zv[comp, , ], nrow = dx[class_ax + 1L])
          sc[[key]][comp, ] <- sc[[key]][comp, ] + nearest_mean_acc(zt_c, zv_c)
        }
      }
    }
    lapply(sc, function(m) m / n_splits)
  }

  # true scores
  message("Computing classification score on data...")
  true_score <- compute_score(X, trialX)

  # shuffled scores
  trialX_shuf  <- trialX    # copy for shuffling
  shuf_scores  <- lapply(keys, function(k) list())
  names(shuf_scores) <- keys

  for (it in seq_len(n_shuffles)) {
    message("\rShuffle ", it, "/", n_shuffles, appendLF = FALSE)

    # shuffle class labels: permute the first parameter axis of trialX
    # (swap slices along dim 3 = first parameter axis after [n_trials, N])
    n_tr <- dim(trialX_shuf)[1]
    perm <- sample(dim(trialX_shuf)[3])
    trialX_shuf <- trialX_shuf[, , perm, , drop = FALSE]

    X_shuf <- apply(trialX_shuf, -1, mean)
    X_shuf <- array(X_shuf, dim = dim(trialX_shuf)[-1])

    sc_shuf <- compute_score(X_shuf, trialX_shuf)

    for (key in keys) {
      shuf_scores[[key]] <- c(shuf_scores[[key]], list(sc_shuf[[key]]))
    }
  }
  message("")

  # build significance masks
  masks <- list()
  for (key in keys) {
    max_shuf      <- Reduce(pmax, shuf_scores[[key]])   # [k, T] element-wise max
    masks[[key]]  <- true_score[[key]] > max_shuf       # [k, T] logical

    # remove isolated significant time-points (n_consecutive filter)
    if (n_consecutive > 1L) {
      for (comp in seq_len(nrow(masks[[key]]))) {
        m   <- masks[[key]][comp, ]
        out <- m
        run <- 0L
        for (t in seq_along(m)) {
          if (m[t]) { run <- run + 1L } else {
            if (run < n_consecutive) out[(t - run):(t - 1L)] <- FALSE
            run <- 0L
          }
        }
        masks[[key]][comp, ] <- out
      }
    }
  }

  masks
}


# ------------------------------------------------------------------------------
# dpca_plot
# ------------------------------------------------------------------------------
# Reproduce the canonical dPCA demo figure (Kobak 2016) using ggplot2.
#
# Panels: columns = marginalizations, rows = components (1..n_comp_show).
# Within each panel: one line per stimulus condition, x = time.
# Assumes the first non-time dimension is the stimulus dimension (dim 2 of Z).
#
# Parameters:
#   Z            : output of dpca_transform (named list of arrays [k, S, T, ...])
#   model        : output of dpca_fit (used for labels and dx)
#   time         : numeric vector length T, default 1:T
#   n_comp_show  : number of components to show per marginalization (default 1)
#   stim_labels  : character vector length S for legend labels (default "s1",..)
#   marg_order   : character vector to set panel column order; default = names(Z)
#   palette      : color palette passed to scale_color_brewer (default "Set1")
#
# Returns: a ggplot object
# ------------------------------------------------------------------------------
dpca_plot <- function(Z, model,
                      time        = NULL,
                      n_comp_show = 1L,
                      stim_labels = NULL,
                      marg_order  = NULL,
                      palette     = "Set1") {

  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("ggplot2 is required. Install it with: install.packages('ggplot2')")

  # --- dimensions -----------------------------------------------------------
  dx <- model$dx                          # e.g. c(S, T) for labels="st"
  S  <- dx[1L]
  T  <- dx[length(dx)]
  if (is.null(time)) time <- seq_len(T)
  if (length(time) != T)
    stop("'time' must have length equal to the time dimension of Z")

  if (is.null(stim_labels)) stim_labels <- paste0("s", seq_len(S))
  if (length(stim_labels) != S)
    stop("'stim_labels' must have length S = ", S)

  marg_names <- if (!is.null(marg_order)) marg_order else names(Z)

  # --- build long data frame ------------------------------------------------
  rows <- list()
  for (marg in marg_names) {
    Zm  <- Z[[marg]]                      # [k, S, T]
    k   <- dim(Zm)[1L]
    n_show <- min(n_comp_show, k)
    for (comp in seq_len(n_show)) {
      for (s in seq_len(S)) {
        rows[[length(rows) + 1L]] <- data.frame(
          marg  = marg,
          comp  = paste0("comp ", comp),
          stim  = stim_labels[s],
          time  = time,
          value = Zm[comp, s, ],
          stringsAsFactors = FALSE
        )
      }
    }
  }
  df <- do.call(rbind, rows)

  # factor ordering for facets
  df$marg <- factor(df$marg, levels = marg_names)
  df$comp <- factor(df$comp, levels = paste0("comp ", seq_len(n_comp_show)))
  df$stim <- factor(df$stim, levels = stim_labels)

  # --- plot ------------------------------------------------------------------
  p <- ggplot2::ggplot(df, ggplot2::aes(
         x = time, y = value, colour = stim, group = stim)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::facet_grid(comp ~ marg, scales = "free_y") +
    ggplot2::scale_colour_brewer(palette = palette, name = "Stimulus") +
    ggplot2::labs(x = "Time", y = "Component activity") +
    ggplot2::theme_classic(base_size = 11) +
    ggplot2::theme(
      strip.background = ggplot2::element_rect(fill = "#f0f0f0", colour = "grey60"),
      strip.text       = ggplot2::element_text(face = "bold"),
      legend.position  = "right"
    )

  p
}