# NoiseTools library
# ------------------------------------------------------------------------------
# Replicate NoiseTools in R http://audition.ens.fr/adc/NoiseTools/
# [Done]
# - nt_fold
# - nt_unfold
# - nt_vecmult
# - nt_normcol
# - nt_mcca
# - nt_mpcarot
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

nt_wmean <- function(x, w = NULL, dim = 1) {
  # y=nt_wmean(x,w,dim) - weighted average
  #
  # y: vector of weighed means
  # 
  # x: column vector or matrix of values to average columnwise
  # w: column vector or matrix of weights (default: all ones)
  # dim: dimension over which to average (default: 1)
  # 
  # if x contains nans the corresponding weights are set to zero
  # original: http://audition.ens.fr/adc/NoiseTools/src/NoiseTools/doc/NoiseTools/nt_wmean.html
  #---------------------------------------------
  if (any(is.na(x))) {
    x[is.na(x)] <- 0
    if (is.null(w)) w <- rep(1, length(x))
    w[is.na(x)] <- 0
  }
  
  if (is.null(w)) {
    return(apply(x, dim, mean))
  } else {
    if (nrow(x) != nrow(w)) stop("Data and weight must have the same number of rows.")
    if (ncol(w) == 1) w <- matrix(rep(w, ncol(x)), nrow = nrow(x))
    if (ncol(w) != ncol(x)) stop("Weight must have the same number of columns as data, or 1.")
    
    return(apply(x * w, dim, sum) / apply(w, dim, sum))
  }
}


nt_fold <- function(x, N) {
  # y=nt_fold(x,N) - fold 2D to 3D
  # 
  # y: 3D matrix of (time * channel * trial)
  # 
  # x: 2D matrix of concatenated data (time * channel)
  # N: number of samples in each trial
  #---------------------------------------------
  if (length(dim(x)) == 3) {
    x <- nt_unfold(x)  # in case it was already folded
  }
  
  if (!is.null(x)) {
    nepochs <- nrow(x) / N
    if (nepochs != round(nepochs)) {
      warning("nsamples not multiple of epoch size, truncating...")
      nepochs <- floor(nepochs)
      x <- x[1:(N * nepochs), ]
    }
    if (nepochs > 1) {
      x <- aperm(array(x, dim = c(N, nrow(x) / N, ncol(x))), c(1, 3, 2))
    }
  }
  
  return(x)
}

nt_unfold <- function(x) {
  # y=nt_unfold(x) - unfold 3D to 2D
  #
  # y: 2D matrix of concatenated data (time * channel)
  #
  # x: 3D matrix of (time * channel * trial)
  #---------------------------------------------
  if (is.null(x)) {
    x <- NULL
  } else {
    dims <- dim(x)
    m <- dims[1]
    n <- dims[2]
    p <- dims[3]
    
    if (!is.na(p)) {
      # If there are multiple trials, flatten the 3D array to 2D
      x <- matrix(aperm(x, c(1, 3, 2)), nrow = m * p, ncol = n)
    } else {
      # If there is only one trial, no need to apply aperm
      x <- matrix(x, nrow = m, ncol = n)
    }
  }
  
  return(x)
}


nt_vecmult <- function(x, v) {
  # y=nt_vecmult(x,v) - multiply all rows or columns of matrix by vector
  # y: product
  #
  # x: 2D matrix
  # v: a vector matching to either 1st or 2nd dim of x
  #---------------------------------------------
  # Check dimensions of x and v
  g(m, n) %=% dim(x)
  o <- NA # not used in the original code
  
  # Ensure that x and v are unfolded (flattened if necessary)
  x <- nt_unfold(x)
  v <- nt_unfold(v)
  
  # Check dimensions of flattened x and v
  g(mm, nn) %=% dim(x)
  g(mv, nv) %=% dim(v) 
  
  # Same number of rows, v should be column vector (or same size as x)
  if (mv==mm){
    # x * column vector v
    if (nv==nn){x <- x * v}
    # x * scaler v
    if (nv==1){x <- x * rep(v, n)}
  
  # Same number of columns, v should be row vector (or same size as x)
  }else if (nv==nn){
    # x * row vector v
    if (mv==mm){x <- x * v}
    
    # x * scaler v 
    if (mv==1){x <- x * rep(v, mm)}
    
  # Something is off
  }else {
    stop('V and X should have same number of rows or columns'); 
  }
  
  # Fold the result back to the original dimensions (if applicable)
  x <- nt_fold(x, m)
  return(x)
}


nt_multishift <- function(x, shifts, pad = 0) {
  # y=nt_multishift(x,shifts,pad) - apply multiple shifts to matrix
  # 
  # y: shifted matrix/vector
  #
  # x: matrix to shift (time X channels)
  # shifts: array of shifts (must be non-negative)
  #
  # X is shifted column by column (all shifts of 1st column, then all
  # shifts of second column, etc).
  #---------------------------------------------
  if (length(shifts) < 1) stop("Shifts array must be provided")
  
  if (length(dim(x)) == 1) {
    n <- length(x)
    z <- matrix(0, n, length(shifts))
    for (i in 1:length(shifts)) {
      z[, i] <- c(rep(0, shifts[i]), x[1:(n - shifts[i])])
    }
  } else if (length(dim(x)) == 2) {
    n <- nrow(x)
    z <- matrix(0, n, ncol(x) * length(shifts))
    for (j in 1:ncol(x)) {
      for (i in 1:length(shifts)) {
        z[, (j - 1) * length(shifts) + i] <- c(rep(0, shifts[i]), x[1:(n - shifts[i]), j])
      }
    }
  }
  
  if (pad) z[n, ] <- 0
  return(z)
}


nt_normcol <- function(x, w = NULL) {
  # y=nt_normcol(x,w) - normalize each column so its weighted msq is 1
  # 
  # y: normalized data
  # norm: vector of norms
  #
  #   x: data to normalize
  #   w: weight
  #
  # If x is 3D, pages are concatenated vertically before calculating the
  # norm. If x is 4D, apply normcol to each book.
  #---------------------------------------------
  
  # If weight is not specified, fill w with 1
  if (is.null(w)) {w <- array(1, dim = dim(x))}
  
  # Four-dimensional case
  if (length(dim(x)) == 4) {
    # Preallocate
    g(m, n, o, p) %=% dim(x)
    y <- array(0, dim = c(m, n, o, p))
    N <- array(0, dim = c(1, n, o))
    
    # Recursive call over 4th dim
    for (k in 1:p) {
      result <- nt_normcol(x[,,,k], w[,,,k])
      y[,,,k] <- result$y
      N <- N + result$norm^2
    }
    return(list(y = y, norm = sqrt(N)))
  }
  
  # Three-dimensional case
  if (length(dim(x)) == 3) {
    # Preallocate
    g(m, n, o) %=% dim(x)
    x <- matrix(x, nrow = m * o, ncol = n)
    w <- matrix(w, nrow = m * o, ncol = n)
    
    # Recursive call
    result <- nt_normcol(x, w)
    y <- array(result$y, dim = c(m, n, o))
    return(list(y = y, norm = result$norm))
  }
  
  # Perform normalization
  N <- colSums((x^2) * w) / colSums(w)
  NN <- ifelse(N > 0, 1 / sqrt(N), 0) # avoids division by 0
  y <- sweep(x, 2, NN, "*")
  
  return(list(y = y, norm = sqrt(N)))
}

# # Test script for nt_normcol function (made by chatGPT)
# 
# set.seed(42)  # For reproducibility
# 
# # 2D Test
# cat("Testing 2D case...\n")
# x_2d <- matrix(rnorm(100), nrow = 10, ncol = 10)  # Random 10x10 matrix
# result_2d <- nt_normcol(x_2d)
# 
# cat("2D Normalization Check (should be ~1):\n")
# print(colMeans(result_2d$y^2))  # Should be approximately 1
# 
# # 3D Test
# cat("\nTesting 3D case...\n")
# x_3d <- array(rnorm(1000), dim = c(10, 10, 10))  # Random 10x10x10 array
# result_3d <- nt_normcol(x_3d)
# 
# cat("3D Normalization Check (should be ~1):\n")
# print(colMeans(matrix(result_3d$y^2, nrow = 100)))  # Flatten and check
# 
# # 4D Test
# cat("\nTesting 4D case...\n")
# x_4d <- array(rnorm(10000), dim = c(10, 10, 10, 10))  # Random 10x10x10x10 array
# result_4d <- nt_normcol(x_4d)
# 
# cat("4D Normalization Check (should be ~1):\n")
# print(colMeans(matrix(result_4d$y^2, nrow = 1000)))  # Flatten and check
# 
# # Weighted 2D Test
# cat("\nTesting 2D case with weights...\n")
# w_2d <- matrix(runif(100, 0.5, 1.5), nrow = 10, ncol = 10)  # Random weights
# result_2d_w <- nt_normcol(x_2d, w_2d)
# 
# cat("2D Weighted Normalization Check (should be ~1):\n")
# print(colSums((result_2d_w$y^2) * w_2d) / colSums(w_2d))  # Should be approximately 1
# 
# # Weighted 3D Test
# cat("\nTesting 3D case with weights...\n")
# w_3d <- array(runif(1000, 0.5, 1.5), dim = c(10, 10, 10))  # Random weights
# result_3d_w <- nt_normcol(x_3d, w_3d)
# 
# cat("3D Weighted Normalization Check (should be ~1):\n")
# print(colSums(matrix((result_3d_w$y^2) * w_3d, nrow = 100)) / colSums(matrix(w_3d, nrow = 100)))
# 
# # Weighted 4D Test
# cat("\nTesting 4D case with weights...\n")
# w_4d <- array(runif(10000, 0.5, 1.5), dim = c(10, 10, 10, 10))  # Random weights
# result_4d_w <- nt_normcol(x_4d, w_4d)
# 
# cat("4D Weighted Normalization Check (should be ~1):\n")
# print(colSums(matrix((result_4d_w$y^2) * w_4d, nrow = 1000)) / colSums(matrix(w_4d, nrow = 1000)))
# 
# cat("\nAll tests completed successfully!\n")

nt_mcca <- function(C, N) {
  #[A,score,AA]=nt_mcca(C,N) - multi-set cca
  #
  # A: global transform matrix
  # score: commonality score (ranges from 1 to N)
  # AA: array of dataset-specific MCCA transform matrices 
  #
  # C: covariance matrix of aggregated data sets
  # N: number of channels of each data set
  # original: http://audition.ens.fr/adc/NoiseTools/src/NoiseTools/doc/NoiseTools/nt_mcca.html
  # Naming of variables follow the original code
  #---------------------------------------------
  if (missing(N) || nrow(C) != ncol(C)) stop("Invalid input: C must be square!")
  if (nrow(C) != round(nrow(C) / N) * N) stop("Dimension mismatch: C is not divisible by N")
  
  # Preallocate: Each X1..n must have equal rows  
  nblocks <- nrow(C) / N
  A <- matrix(0, nrow(C), ncol(C))
  
  # Sphere by blocks 
  for (iBlock in 1:nblocks) {
    # Access parts of covariance matrix
    idx <- ((iBlock - 1) * N + 1):(iBlock * N)
    CC <- C[idx, idx]

    # First PCA via svd
    g(S, V, v) %=% svd(CC) # d,u,v
    idx2 <- order(S, decreasing = TRUE)
    topcs <- V[, idx2]
    
    # For numerical stability
    E <- 1 - 10^(-12) # very close to 1
    S <- S^E # exponentiate
    EE <- 1 / S # inverse of singular values
    EE[EE <= 0] <- 0 # in case S being super close to 0
    
    # Within-block "whitening" (decorrelation & equal variance normalization)
    # Update A using principal components (topcs) and a normalization transformation (sqrt(EE))
    # topcs <- V[, idx2]: eigenvectors of CC (a block of covariance matrix)
    # diag(sqrt(EE)): a diagonal scaling matrix
    A[idx, idx] <- topcs %*% diag(sqrt(EE))
  }
  
  # Linear transformation of the covariance matrix C using the matrix A
  # Pre-multiplication by t(A):t(A) %*% C: transforms C into a new basis defined by 
  # Post-multiplication by A: ensures that the transformation is applied symmetrically
  C <- t(A) %*% C %*% A
  
  # Second PCA via svd
  g(S, V, v) %=% svd(C) # d,u,v
  idx <- order(S, decreasing = TRUE)
  topcs <- V[, idx]
  
  # Global transform matrix ("Y" in de Cheveigne et al, 2019)
  # To get V1..n,  v <- x %*% A, where x=(original data used for C)
  A <- A %*% topcs
  
  # Dataset-specific transformation matrix (Y1...n in in de Cheveigne et al, 2019)
  AA <- list()
  for (iBlock in 1:nblocks) {
    AA[[iBlock]] <- A[((iBlock - 1) * N + 1):(iBlock * N), , drop = FALSE]
  }

  # Commonality scores ("Sigma" in de Cheveigne et al, 2019)
  C <- t(topcs) %*% C %*% topcs
  score <- diag(C)
  #score <- pmin(pmax(diag(C), 0), nblocks) # fix numerical errors1
  score <- pmax(diag(C), 0) # fix numerical errors2
  score <- score / max(score) * nblocks
  return(list(A = A, score = score, AA = AA))
}



nt_pcarot <- function(cov, nkeep = NULL, threshold = NULL, N = NULL) {
  # [topcs, eigenvalues] = pcarot(cov, nkeep, threshold, N) - PCA matrix from covariance
  #
  #  topcs: PCA rotation matrix
  #  eigenvalues: PCA eigenvalues
  #  
  #  cov: covariance matrix
  #  nkeep: number of components to keep
  #  threshold: discard components below this threshold
  #  N: eigs' K parameter (if absent: use eig)
  #---------------------------------------------
  
  # Perform SVD decomposition
  g(S, V, v) %=% svd(CC) # d,u,v
  
  # Sort eigenvalues and eigenvectors in descending order
  idx <- order(S, decreasing = TRUE)
  eigenvalues <- S[idx]
  topcs <- V[, idx]
  
  # Truncate based on threshold
  if (!is.null(threshold)) {
    valid_indices <- which(eigenvalues / eigenvalues[1] > threshold)
    topcs <- topcs[, valid_indices, drop = FALSE]
    eigenvalues <- eigenvalues[valid_indices]
  }
  
  # Keep only a certain number of components
  if (!is.null(nkeep)) {
    nkeep <- min(nkeep, ncol(topcs))
    topcs <- topcs[, 1:nkeep, drop = FALSE]
    eigenvalues <- eigenvalues[1:nkeep]
  }
  
  return(list(topcs = topcs, eigenvalues = eigenvalues))
}


nt_pca <- function(x, shifts = 0, nkeep = NULL, threshold = NULL, w = NULL) {
  # [z, idx] = nt_pca(x, shifts, nkeep, threshold, w) - time-shift PCA
  #
  #  z: principal components (PCs)
  #  idx: x[idx] maps to z
  #
  #  x: data matrix
  #  shifts: array of shifts to apply
  #  nkeep: number of components to keep
  #  threshold: discard PCs with eigenvalues below this
  #  w: weights
  #---------------------------------------------
  if (!is.numeric(x)) stop("Input x must be numeric")
  
  # Check dimensions
  dims <- dim(x)
  g(m, n, o) %=% dims
  if (length(dims) == 2) {o <- 1}
  if (!(length(dims) %in% c(2,3))) {stop("Unsupported data structure for x")}
  
  # Offset of z relative to x
  offset <- max(0, -min(shifts))
  shifts <- shifts + offset  # Adjust shifts to be non-negative
  idx <- offset + (1:(m - max(shifts)))  # x[idx] maps to z
  
  # Compute covariance
  c <- nt_cov(x, shifts, w)
  
  # Compute PCA matrix
  pca_result <- nt_pcarot(c, nkeep, threshold)
  topcs <- pca_result$topcs
  
  # Apply PCA matrix to time-shifted data
  z <- array(0, dim = c(length(idx), ncol(topcs), o))
  for (k in 1:o) {
    z[, , k] <- nt_multishift(x[, , k], shifts) %*% topcs
  }
  
  return(list(z = z, idx = idx))
}



