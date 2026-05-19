# demixedPCA_lib.R
# ==============================================================================
# R implementation of Kobak et al. (2016) demixed PCA
# Python original: https://github.com/machenslab/dPCA
#
# Functions:
#   dpca_get_marginalizations -- enumerate all 2^K - 1 parameter subsets
#   dpca_marginalize          -- ANOVA decomposition of X into orthogonal marginals
#   dpca_fit                  -- solve for encoder P and decoder D per marginalization
#   dpca_transform            -- project data: Z_phi = D_phi^T X_flat
#   dpca_inverse_transform    -- reconstruct: X_recon = P_phi Z_phi
#   dpca_reconstruct          -- fit_transform + inverse in one call
#   dpca_significance         -- shuffle-test significance per component and timepoint
#   dpca_plot                 -- full-figure plot (MATLAB dpca_plot equivalent)
#
# Data convention:
#   X      : array [N, d1, d2, ...], N = neurons/channels, d1..dK = parameter dims
#   labels : character string, one char per parameter axis (e.g. "st")
#   Both trial-averaged; centering (subtract per-neuron grand mean) is done
#   inside dpca_fit and dpca_transform.
#
# Reference:
#   Kobak D, Brendel W, Constantinidis C, Feierstein CE, Kepecs A, Mainen ZF,
#   Qi XL, Romo R, Uchida N, Machens CK (2016). Demixed principal component
#   analysis of neural population data. eLife 5:e10989.
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
# prep_cond_avg: CONDITION-AVERAGED DATA PER BALANCE FOLD
# =========================================================================
# [Role]:
#   Compute per-balance-fold condition averages from trial-by-trial data,
#   returning a list of [N x T] matrices ready for dpca_fit. Used in
#   dpca_significance to create train/test splits that respect balance.
#
# [Inputs]:
#   d      : array [n_trial x n_time x N], trial-by-trial neural data.
#            n_trial = trials, n_time = time-points, N = neurons.
#   blIDX  : integer vector [n_trial], balance fold ID per trial
#            (e.g. ds_bl[["balanceID"]]).
#   grpIDX : character/integer vector [n_trial], condition label per trial
#            (e.g. ds_bl[[modelV]]).
#
# [Outputs]:
#   XnL : list of matrices [N x T], one element per balance fold.
#         Each matrix is the condition-averaged activity for that fold.
#
# [Algorithm]:
#   1. Average d over trials within each (balanceID, condition) cell (narray::map).
#   2. Sort rows, split by balanceID (narray::split).
#   3. Transpose each slice to [N x T] format.
#
# [Notes]:
#   - Requires the narray package.
#   - dimAdj is a project-local helper assumed to be in scope.
# =========================================================================
prep_cond_avg <- function(d, blIDX, grpIDX) {
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

# =========================================================================
# dpca_get_marginalizations: ENUMERATE ALL 2^K - 1 PARAMETER SUBSETS
# =========================================================================
# [Role]:
#   Build the complete powerset of K parameter axes (excluding the empty set),
#   returning a named list of 0-based integer index vectors. This set of
#   subsets defines the marginalizations that dPCA decomposes variance into:
#   each marginalization corresponds to one ANOVA-style effect (main effects
#   and all interactions).
#
# [Inputs]:
#   labels : character string, one character per parameter axis (e.g. "st").
#            Length K determines the 2^K - 1 output elements.
#
# [Outputs]:
#   Named list of integer vectors, e.g. for labels="st":
#     list(s = 0L, t = 1L, st = c(0L, 1L))
#   Indices are 0-based parameter axis indices (i.e. dim 2, 3, ... of X,
#   where dim 1 is the neuron axis N).
#
# [Algorithm]:
#   For r = 1..K, enumerate all length-r combinations of 1..K via combn.
#   Key = paste of label characters; value = sorted 0-based axis indices.
#
# [Notes]:
#   - 0-based indices match Python/MATLAB conventions used in ARCHITECTURE.
#   - For K=2 ("st"): 3 elements. For K=3 ("stc"): 7 elements.
#   - Order within the list is lexicographic by subset size then content.
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989. Appendix.
# =========================================================================
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


# =========================================================================
# dpca_marginalize: ANOVA DECOMPOSITION INTO ORTHOGONAL MARGINALS
# =========================================================================
# [Role]:
#   Decompose the centered data array X into orthogonal ANOVA-style
#   marginalizations. Each marginalization captures the variance attributable
#   to one parameter combination (main effects and interactions).
#
#   Example (labels="st", X=[N,S,T]):
#     X_s[n,s,t] = mean_t(Xc[n,s,:])     stimulus effect (constant over t)
#     X_t[n,s,t] = mean_s(Xc[n,:,t])     time effect    (constant over s)
#     X_st        = Xc - X_s - X_t        interaction
#     Sum: X_s + X_t + X_st = Xc   (exact orthogonal partition)
#
# [Inputs]:
#   X      : array [N, d1, d2, ...], trial-averaged neural data.
#            N = neurons, d1..dK = parameter dimensions.
#   labels : character string, one char per parameter axis.
#
# [Outputs]:
#   Named list of matrices [N, prod(d1, ..., dK)], one per marginalization.
#   Columns are the flattened parameter grid in row-major order.
#
# [Algorithm]:
#   Follows Kobak et al. (2016) Appendix, inclusion-exclusion:
#   1. Center X: subtract per-neuron grand mean -> Xc.
#   2. For each key phi (sorted parameter subset):
#      pre_mean[[phi]] = mean of Xc over the axes listed in phi
#      (singleton shape at those axes, full shape elsewhere).
#   3. base = pre_mean[[complement(phi)]] = mean over axes NOT in phi.
#   4. Subtract all strict sub-marginals of phi (inclusion-exclusion)
#      to remove overlap and ensure orthogonality.
#   5. Expand singletons to full dX; flatten to [N, prod(dx)].
#
# [Notes]:
#   - expand_to is an internal broadcast helper (R lacks implicit broadcasting).
#   - pre_mean is built incrementally: each multi-char key's pre_mean is the
#     parent key's pre_mean averaged over the last new axis (chain reduction).
#   - The orthogonality guarantee: sum of all marginals = Xc exactly.
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989. Appendix.
# =========================================================================
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


# =========================================================================
# dpca_fit: SOLVE FOR ENCODER P AND DECODER D PER MARGINALIZATION
# =========================================================================
# [Role]:
#   Fit a dPCA model by finding, for each marginalization phi, a k-dimensional
#   encoder P_phi and decoder D_phi such that:
#     - D_phi captures variance specific to phi (demixed from other effects).
#     - P_phi reconstructs back to the full neural space.
#   Unlike PCA, P != D: the decoder is specialized to one parameter combination
#   while the encoder spans the full population. This asymmetry is what makes
#   the decomposition "demixed."
#
#   Kobak et al. (2016): "dPCA finds components that each mostly capture the
#   variance of just one marginalization."
#
# [Inputs]:
#   X           : array [N, d1, d2, ...], trial-averaged neural data.
#   labels      : character string, one char per parameter axis.
#   n_components: integer or named list per marginalization (default 10).
#                 Number of components to extract per marginalization.
#   regularizer : numeric >= 0, ridge penalty = regularizer * var(X).
#                 Set 0 for no regularization.
#
# [Outputs]:
#   dpca_model list with:
#   $P     : named list of matrices [N x k], encoders (one per marginalization)
#   $D     : named list of matrices [N x k], decoders (one per marginalization)
#   $labels: character string
#   $dx    : integer vector, parameter dimensions
#   $N     : integer, number of neurons
#   $margs : marginalization list from dpca_get_marginalizations
#   $n_components, $regularizer : metadata
#
# [Algorithm]:
#   For each marginalization phi (Kobak et al. 2016, Appendix):
#     C_phi = X_phi_flat %*% pinv(X_flat)           [N x N]
#     Fact  = C_phi %*% X_flat                       [N x prod(dx)]
#     SVD of Fact (truncated to k components):
#       P_phi = U                                    [N x k]  (encoder)
#       D_phi = C_phi^T %*% U                        [N x k]  (decoder)
#   With regularization: append lambda*I to X and zeros to each X_phi
#   (standard Tikhonov trick: argmin ||X_phi - D P^T X||^2 + lam||P||^2).
#
# [Notes]:
#   - P is the left singular vectors of Fact (optimal reconstruction direction).
#   - D = C_phi^T P is the projection direction specialized to phi.
#   - In standard PCA, C_phi = I and D = P.
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989. Appendix.
# =========================================================================
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


# =========================================================================
# dpca_transform: PROJECT DATA ONTO dPCA COMPONENTS
# =========================================================================
# [Role]:
#   Project the neural data array X onto the dPCA decoder directions found
#   during dpca_fit, producing low-dimensional latent trajectories for each
#   marginalization. Also computes explained variance per component.
#
# [Inputs]:
#   X     : array [N, d1, d2, ...], same shape as used in dpca_fit.
#   model : dpca_model list output of dpca_fit.
#
# [Outputs]:
#   Named list of arrays [k, d1, d2, ...], one per marginalization.
#   The list carries an attribute "explained_variance_ratio": a named list
#   of numeric vectors, one per marginalization, giving the fraction of
#   total variance explained by each component.
#
# [Algorithm]:
#   1. Flatten X to [N, prod(dx)]; center per neuron.
#   2. For each marginalization phi:
#        Z_flat = D_phi^T X_flat          [k x prod(dx)]
#        Z      = array(Z_flat, [k, dx])  [k, d1, d2, ...]
#        EV[i]  = sum(Z_flat[i,]^2) / total_var
#
# [Notes]:
#   - Uses D_phi (decoder), not P_phi (encoder). This is the projection
#     that is specialized to marginalization phi.
#   - Explained variance is relative to total (all-marginalization) variance.
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989. Appendix.
# =========================================================================
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


# =========================================================================
# dpca_inverse_transform: RECONSTRUCT FROM LATENT COMPONENTS
# =========================================================================
# [Role]:
#   Reconstruct the full neural array [N, d1, ...] from the latent components
#   Z_phi of one marginalization, using the encoder P_phi. Combines with
#   dpca_transform to give the dPCA-filtered version of the data for a single
#   parameter effect.
#
# [Inputs]:
#   Z               : array [k, d1, d2, ...], latent components for one
#                     marginalization (one element of dpca_transform output).
#   model           : dpca_model list output of dpca_fit.
#   marginalization : character key selecting which marginalization (e.g. "s").
#
# [Outputs]:
#   array [N, d1, d2, ...], reconstructed neural data for marginalization phi.
#
# [Algorithm]:
#   X_recon = P_phi %*% Z_flat    where Z_flat = matrix(Z, nrow=k)   [N x prod(dx)]
#   Reshape to [N, d1, d2, ...].
#
# [Notes]:
#   - Uses P_phi (encoder), not D_phi (decoder). The reconstruction is in the
#     full neural space.
#   - dpca_reconstruct(X, model, phi) = dpca_inverse_transform(
#       dpca_transform(X, model)[[phi]], model, phi).
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989. Appendix.
# =========================================================================
dpca_inverse_transform <- function(Z, model, marginalization) {
  Pk    <- model$P[[marginalization]]                 # [N × k]
  dx    <- model$dx
  k     <- ncol(Pk)
  Zflat <- matrix(Z, nrow = k)                       # [k × prod(dx)]

  Xrec  <- Pk %*% Zflat                              # [N × prod(dx)]
  array(Xrec, dim = c(model$N, dx))
}


# =========================================================================
# dpca_reconstruct: CONVENIENCE TRANSFORM + INVERSE FOR ONE MARGINALIZATION
# =========================================================================
# [Role]:
#   Shorthand for dpca_inverse_transform(dpca_transform(X, model)[[phi]],
#   model, phi). Returns the dPCA-filtered reconstruction of X for a single
#   parameter effect -- the component of population activity attributable
#   to that effect alone.
#
# [Inputs]:
#   X               : array [N, d1, d2, ...], same shape as used in dpca_fit.
#   model           : dpca_model list output of dpca_fit.
#   marginalization : character key (e.g. "s", "t", "st").
#
# [Outputs]:
#   array [N, d1, d2, ...], reconstructed neural data for the given marginalization.
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989.
# =========================================================================
dpca_reconstruct <- function(X, model, marginalization) {
  Z <- dpca_transform(X, model)[[marginalization]]
  dpca_inverse_transform(Z, model, marginalization)
}


# =========================================================================
# dpca_significance: SHUFFLE-TEST SIGNIFICANCE PER COMPONENT AND TIMEPOINT
# =========================================================================
# [Role]:
#   Assess statistical significance of each dPCA component at each time point
#   by comparing classification accuracy (nearest-mean classifier in latent
#   space) on real data against a shuffled-label null distribution.
#   Returns a logical mask [k x T] that is TRUE where the component is
#   significant (true accuracy > maximum shuffle accuracy).
#
# [Inputs]:
#   X            : array [N, d1, ..., T], trial-averaged neural data.
#   trialX       : array [n_trials, N, d1, ..., T], trial-by-trial data.
#   model        : dpca_model from dpca_fit, or NULL to re-fit inside.
#   labels       : character string (used to fit if model is NULL).
#   n_shuffles   : integer, number of label shuffles for null distribution
#                  (p-value threshold ~ 1/n_shuffles; default 100).
#   n_splits     : integer, train-test splits per evaluation (default 100).
#   n_consecutive: require this many consecutive significant time-points
#                  before calling a run significant (default 1 = no filtering).
#   axis         : integer, parameter axis (1-based) over which to evaluate
#                  significance; typically the time axis (last parameter axis).
#   n_components : passed to dpca_fit if model is NULL (default 10).
#
# [Outputs]:
#   Named list of logical matrices [k x T], one per tested marginalization.
#   TRUE at position [comp, t] means component comp is significant at time t.
#
# [Algorithm]:
#   1. For n_splits: randomly hold out one trial, fit dPCA on the rest,
#      project held-out trial, classify using nearest-mean in latent space.
#      Accumulate mean classification accuracy per component and time point.
#   2. Repeat with class labels shuffled (n_shuffles times) to build the null.
#   3. Significance: true accuracy > max(shuffled accuracies) at each [comp, t].
#   4. Apply n_consecutive filter: isolated significant time-points are removed.
#
# [Notes]:
#   - Excludes the time-only marginalization (cannot classify over the axis
#     used to define classes).
#   - "Nearest-mean" classifier: for each condition, find the closest class
#     mean in 1D latent space; accuracy = fraction correct.
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989. Supplementary Methods.
# =========================================================================
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


# =========================================================================
# dpca_plot: FULL-FIGURE dPCA PLOT (MATLAB dpca_plot EQUIVALENT)
# =========================================================================
# [Role]:
#   Produce a publication-ready summary figure of a fitted dPCA model,
#   equivalent to the MATLAB dpca_plot() from Kobak et al. (2016).
#   Assembles component time-series panels in a marginalization x component
#   grid, with optional explained-variance panels in a left column.
#
# [Inputs]:
#   Z            : output of dpca_transform (named list of arrays [k x S x T]).
#   model        : dpca_model list output of dpca_fit.
#   time         : numeric vector length T (default 1:T).
#   n_comp_show  : integer, components per marginalization to show (default 3).
#   stim_labels  : character [S], legend labels (default "s1","s2",...).
#   marg_order   : character vector, display order of marginalizations
#                  (default = names(Z)).
#   marg_colours : named character [M], hex/R colour per marginalization.
#   marg_names   : named character [M], display name per marginalization.
#   time_events  : numeric, x positions for vertical dashed event markers.
#   signif       : named list of logical matrices [k x T] from dpca_significance;
#                  significant time windows are shaded in the marginalization colour.
#   show_ev      : logical, show left-column EV panels (default TRUE).
#   palette      : RColorBrewer palette name for stimulus colours (default "Set1").
#
# [Outputs]:
#   If patchwork is installed: a patchwork object (print to display).
#   Otherwise: a named list of individual ggplot2 objects.
#
# [Layout]:
#   Main grid  : M rows x n_comp_show columns of time-series panels.
#                Each panel = one component of one marginalization.
#                Lines = stimulus conditions; colour = marg_colour border.
#   Left column (if show_ev = TRUE and explained variance available):
#     - Cumulative EV curve (top)
#     - Stacked bar: EV per component, colour = marginalization (middle)
#     - Pie chart: total EV per marginalization (bottom)
#
# [Notes]:
#   - Requires ggplot2; patchwork is optional (for automatic layout assembly).
#   - Border colour of each panel indicates its marginalization.
#   - Significance shading uses alpha=0.15 fill in the marginalization colour.
#
# [Reference]:
#   Kobak et al. (2016) eLife 5:e10989. Fig. 2 and Supplementary Fig. 1.
# =========================================================================
dpca_plot <- function(Z, model,
                      time         = NULL,
                      n_comp_show  = 3L,
                      stim_labels  = NULL,
                      marg_order   = NULL,
                      marg_colours = NULL,
                      marg_names   = NULL,
                      time_events  = NULL,
                      signif       = NULL,
                      show_ev      = TRUE,
                      palette      = "Set1") {

  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("ggplot2 is required. Install it with: install.packages('ggplot2')")

  g <- ggplot2::ggplot   # shorthand

  # ---------------------------------------------------------------------------
  # 0. Dimensions and defaults
  # ---------------------------------------------------------------------------
  dx <- model$dx
  S  <- dx[1L]
  T  <- dx[length(dx)]
  if (is.null(time)) time <- seq_len(T)
  if (length(time) != T)
    stop("'time' must have length equal to the time dimension of Z (", T, ")")

  marg_keys <- if (!is.null(marg_order)) marg_order else names(Z)
  M <- length(marg_keys)

  if (is.null(stim_labels)) stim_labels <- paste0("s", seq_len(S))

  # marginalization display names
  if (is.null(marg_names)) {
    mn <- setNames(marg_keys, marg_keys)
  } else {
    mn <- setNames(as.character(marg_names), marg_keys)
  }

  # marginalization colours (one per key)
  default_marg_cols <- c(
    "#4393C3", "#D6604D", "#74C476", "#9E7EC8",
    "#F4A582", "#A6D96A", "#FDB863", "#B2ABD2"
  )
  if (is.null(marg_colours)) {
    mc <- setNames(default_marg_cols[seq_len(M)], marg_keys)
  } else {
    mc <- setNames(as.character(marg_colours), marg_keys)
  }

  # explained variance per component (from dpca_transform attribute)
  ev_list <- attr(Z, "explained_variance_ratio")

  # ---------------------------------------------------------------------------
  # helper: theme for component panels
  # ---------------------------------------------------------------------------
  theme_comp <- function(show_x = TRUE, show_y = TRUE) {
    t <- ggplot2::theme_classic(base_size = 10) +
      ggplot2::theme(
        plot.margin   = ggplot2::margin(2, 4, 2, 4),
        legend.position = "none"
      )
    if (!show_x) t <- t + ggplot2::theme(
      axis.text.x  = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_blank(),
      axis.title.x = ggplot2::element_blank()
    )
    if (!show_y) t <- t + ggplot2::theme(
      axis.text.y  = ggplot2::element_blank(),
      axis.ticks.y = ggplot2::element_blank(),
      axis.title.y = ggplot2::element_blank()
    )
    t
  }

  # ---------------------------------------------------------------------------
  # 1. Component time-series panels
  # ---------------------------------------------------------------------------
  stim_fct <- factor(stim_labels, levels = stim_labels)

  panels <- list()
  for (mi in seq_len(M)) {
    marg <- marg_keys[mi]
    Zm   <- Z[[marg]]             # [k, S, T]
    k    <- dim(Zm)[1L]
    n_show <- min(n_comp_show, k)
    ev_k   <- if (!is.null(ev_list)) ev_list[[marg]] else NULL

    for (ci in seq_len(n_show)) {
      # significance shading region data
      shade_df <- NULL
      if (!is.null(signif) && !is.null(signif[[marg]])) {
        sig_row <- signif[[marg]][ci, ]
        if (any(sig_row)) {
          rle_sig  <- rle(sig_row)
          ends     <- cumsum(rle_sig$lengths)
          starts   <- ends - rle_sig$lengths + 1L
          runs     <- which(rle_sig$values)
          shade_df <- data.frame(
            xmin = time[starts[runs]],
            xmax = time[ends[runs]]
          )
        }
      }

      # line data
      line_rows <- vector("list", S)
      for (s in seq_len(S)) {
        line_rows[[s]] <- data.frame(
          time  = time,
          value = Zm[ci, s, ],
          stim  = stim_fct[s],
          stringsAsFactors = FALSE
        )
      }
      df <- do.call(rbind, line_rows)
      df$stim <- factor(df$stim, levels = stim_labels)

      # panel title
      ev_pct <- if (!is.null(ev_k)) sprintf(" [%.1f%%]", ev_k[ci] * 100) else ""
      ptitle <- sprintf("%s comp %d%s", mn[[marg]], ci, ev_pct)

      p <- g(df, ggplot2::aes(x = time, y = value,
                               colour = stim, group = stim)) +
        ggplot2::ggtitle(ptitle) +
        ggplot2::geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3)

      # significance shading
      if (!is.null(shade_df) && nrow(shade_df) > 0) {
        ymax_val <- max(abs(df$value), na.rm = TRUE) * 1.1
        p <- p + ggplot2::geom_rect(
          data = shade_df,
          ggplot2::aes(xmin = xmin, xmax = xmax,
                       ymin = -Inf, ymax = Inf),
          fill = mc[[marg]], alpha = 0.15,
          inherit.aes = FALSE
        )
      }

      # time events
      if (!is.null(time_events)) {
        ev_df <- data.frame(xint = time_events)
        p <- p + ggplot2::geom_vline(
          data = ev_df,
          ggplot2::aes(xintercept = xint),
          colour = "grey60", linewidth = 0.4, linetype = "dashed"
        )
      }

      # lines
      p <- p +
        ggplot2::geom_line(linewidth = 0.7) +
        ggplot2::scale_colour_brewer(palette = palette, name = "Stimulus") +
        ggplot2::labs(x = "Time", y = "Activity") +
        # marginalization colour strip on right side
        ggplot2::theme_classic(base_size = 10) +
        ggplot2::theme(
          plot.title      = ggplot2::element_text(size = 8, hjust = 0.5,
                                                  colour = mc[[marg]],
                                                  face = "bold"),
          plot.background = ggplot2::element_rect(
            fill = "white",
            colour = mc[[marg]], linewidth = 1.2
          ),
          plot.margin     = ggplot2::margin(3, 5, 3, 5),
          legend.position = if (mi == 1 && ci == 1) "right" else "none",
          axis.text.x  = if (mi < M) ggplot2::element_blank() else ggplot2::element_text(),
          axis.ticks.x = if (mi < M) ggplot2::element_blank() else ggplot2::element_line(),
          axis.title.x = if (mi < M) ggplot2::element_blank() else ggplot2::element_text(),
          axis.text.y  = if (ci > 1)  ggplot2::element_blank() else ggplot2::element_text(),
          axis.ticks.y = if (ci > 1)  ggplot2::element_blank() else ggplot2::element_line(),
          axis.title.y = if (ci > 1)  ggplot2::element_blank() else ggplot2::element_text()
        )

      panels[[paste0(marg, "_", ci)]] <- p
    }
  }

  # ---------------------------------------------------------------------------
  # 2. EV panels (left column)
  # ---------------------------------------------------------------------------
  ev_panels <- list()

  if (show_ev && !is.null(ev_list)) {

    # build long data frame of all per-component EVs
    ev_rows <- list()
    for (marg in marg_keys) {
      ev_k <- ev_list[[marg]]
      n_show <- min(n_comp_show, length(ev_k))
      for (ci in seq_len(n_show)) {
        ev_rows[[length(ev_rows) + 1L]] <- data.frame(
          marg     = marg,
          comp_lbl = paste0(mn[[marg]], ci),
          comp_idx = (match(marg, marg_keys) - 1L) * n_comp_show + ci,
          ev_pct   = ev_k[ci] * 100,
          stringsAsFactors = FALSE
        )
      }
    }
    ev_df <- do.call(rbind, ev_rows)
    ev_df$comp_lbl <- factor(ev_df$comp_lbl,
                              levels = ev_df$comp_lbl[order(ev_df$comp_idx)])
    ev_df$marg     <- factor(ev_df$marg, levels = marg_keys)

    # 2a. stacked bar (one column per component, fill = marg colour)
    bar_colours <- setNames(mc[marg_keys], marg_keys)
    p_bar <- g(ev_df, ggplot2::aes(x = comp_lbl, y = ev_pct, fill = marg)) +
      ggplot2::geom_col(width = 0.75) +
      ggplot2::scale_fill_manual(values = bar_colours, guide = "none") +
      ggplot2::labs(x = "Component", y = "Variance explained (%)") +
      ggplot2::theme_classic(base_size = 9) +
      ggplot2::theme(
        axis.text.x  = ggplot2::element_text(angle = 45, hjust = 1, size = 7),
        plot.margin  = ggplot2::margin(4, 4, 4, 4)
      )
    ev_panels[["bar"]] <- p_bar

    # 2b. cumulative EV
    cum_rows <- list()
    running  <- 0
    for (marg in marg_keys) {
      ev_k <- ev_list[[marg]]
      n_show <- min(n_comp_show, length(ev_k))
      for (ci in seq_len(n_show)) {
        running <- running + ev_k[ci] * 100
        cum_rows[[length(cum_rows) + 1L]] <- data.frame(
          idx    = length(cum_rows) + 1L,
          cumev  = running,
          marg   = marg,
          stringsAsFactors = FALSE
        )
      }
    }
    cum_df <- do.call(rbind, cum_rows)
    cum_df$marg <- factor(cum_df$marg, levels = marg_keys)

    p_cum <- g(cum_df, ggplot2::aes(x = idx, y = cumev)) +
      ggplot2::geom_line(colour = "grey30", linewidth = 0.8) +
      ggplot2::geom_point(ggplot2::aes(colour = marg), size = 2.5) +
      ggplot2::scale_colour_manual(values = mc, guide = "none") +
      ggplot2::scale_y_continuous(limits = c(0, 100)) +
      ggplot2::labs(x = "Component", y = "Cumulative EV (%)") +
      ggplot2::theme_classic(base_size = 9) +
      ggplot2::theme(plot.margin = ggplot2::margin(4, 4, 4, 4))
    ev_panels[["cum"]] <- p_cum

    # 2c. pie chart (total EV per marginalization)
    pie_rows <- lapply(marg_keys, function(m) {
      data.frame(marg = m, total = sum(ev_list[[m]]) * 100,
                 stringsAsFactors = FALSE)
    })
    pie_df <- do.call(rbind, pie_rows)
    pie_df$marg  <- factor(pie_df$marg, levels = marg_keys)
    pie_df$label <- sprintf("%s\n%.0f%%", mn[marg_keys], pie_df$total)

    p_pie <- g(pie_df, ggplot2::aes(x = "", y = total,
                                     fill = marg, label = label)) +
      ggplot2::geom_col(width = 1, colour = "white") +
      ggplot2::geom_text(position = ggplot2::position_stack(vjust = 0.5),
                         size = 2.8) +
      ggplot2::coord_polar("y") +
      ggplot2::scale_fill_manual(values = mc, guide = "none") +
      ggplot2::labs(x = NULL, y = NULL,
                    title = "Total EV\nper factor") +
      ggplot2::theme_void(base_size = 9) +
      ggplot2::theme(
        plot.title  = ggplot2::element_text(size = 8, hjust = 0.5),
        plot.margin = ggplot2::margin(4, 4, 4, 4)
      )
    ev_panels[["pie"]] <- p_pie
  }

  # ---------------------------------------------------------------------------
  # 3. Assemble with patchwork (if available) or return list
  # ---------------------------------------------------------------------------
  all_panels <- c(panels, ev_panels)

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("Install patchwork for automatic layout: install.packages('patchwork')")
    return(invisible(all_panels))
  }
  suppressPackageStartupMessages(library(patchwork, warn.conflicts = FALSE))

  # build main grid: M rows × n_comp_show columns
  grid_plots <- list()
  for (mi in seq_len(M)) {
    marg   <- marg_keys[mi]
    n_show <- min(n_comp_show, dim(Z[[marg]])[1L])
    for (ci in seq_len(n_comp_show)) {
      key <- paste0(marg, "_", ci)
      grid_plots[[length(grid_plots) + 1L]] <-
        if (ci <= n_show) panels[[key]] else patchwork::plot_spacer()
    }
  }

  main_grid <- patchwork::wrap_plots(grid_plots,
                                     ncol = n_comp_show, nrow = M)

  if (length(ev_panels) == 3) {
    left_col <- (ev_panels[["cum"]] / ev_panels[["bar"]] / ev_panels[["pie"]]) +
      patchwork::plot_layout(heights = c(1, 1, 1))

    result <- left_col | main_grid
    result <- result + patchwork::plot_layout(widths = c(1, n_comp_show))
  } else {
    result <- main_grid
  }

  result
}