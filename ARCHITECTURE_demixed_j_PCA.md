# demixed_jPCA: Master Document
### Architecture + Implementation Plan (R)

> **Prerequisite: How to Use This Document**
> Before writing any code, always open and read the corresponding section of the paper.
> Every design choice includes a citation where available.
> Only move to the code after you can explain in your own words *why* this step is necessary
> and *why* this implementation choice was made.
> **Answer the comprehension check questions before advancing to the next step.**
> This document is language-specific: all implementations are in R, following the noiseTools/MCCA coding conventions.

---

## 1. Project Goals

0. **The goal is understanding**: replicate dPCA and jPCA step by step in R — at the level where every line maps to a specific equation in the paper
1. Implement **demixed PCA (dPCA)** in R: decompose EEG population activity into task-parameter-specific subspaces (time, rule, stimulus, rule×stimulus)
2. Implement **jPCA** in R: detect rotational dynamics within a given subspace
3. **Connect dPCA → jPCA**: apply jPCA within each dPCA-defined subspace to ask whether dynamics in the rule subspace (or stimulus subspace) are rotational
4. Design the entire pipeline around **existing EEG decoding input format** so that dPCA/jPCA become drop-in analyses alongside LDA decoding
5. **Always write a test against the Python reference**: for every function, verify numerical output matches `machenslab/dPCA` (Python) on the same toy data

### Three Research Goals

**Goal 1 — Task-parameter demixing**

Standard PCA on population EEG mixes rule, stimulus, and time variance into each component. dPCA separates them:

```
If rule and stimulus are mixed in PC1
→ PCA cannot tell whether dynamics are rule-driven or stimulus-driven

dPCA → rule subspace Z_r, stimulus subspace Z_s, interaction Z_rs
→ Each subspace captures variance attributable to exactly one factor (or interaction)
→ Enables factor-specific interpretation of temporal dynamics
```

**Goal 2 — Rotational dynamics within task subspaces**

jPCA asks whether population dynamics follow a rotation (ẋ ≈ M_skew x):

```
jPCA on full data   → rotation mixes all task factors (uninterpretable)
jPCA on Z_r (rule subspace from dPCA)
  → Does the rule representation rotate over time?
  → Is the rotation specific to rule-switching, not stimulus change?

Key question: is there rotation in Z_r that is not present in Z_s?
→ Dissociates rule dynamics from stimulus dynamics geometrically
```

**Goal 3 — Connection to EmbeddingRNN geometry**

EmbeddingRNN predicts that conjunction necessity steepens the gradient landscape, producing stronger rule subspace structure. dPCA quantifies this empirically:

```
High conjunction necessity → rule subspace variance > stimulus subspace variance
Low conjunction necessity  → variance more evenly distributed across subspaces

dPCA variance ratio (rule / stimulus) ↔ EmbeddingRNN gradient steepness
jPCA rotation amplitude in Z_r ↔ RNN hidden state rotation at rule transitions
```

### Key Mathematical Intuition

```
Standard PCA:
  maximize total variance → components mix all task factors

dPCA:
  for each subset φ ⊆ {rule, stim, rule×stim, time}:
    compute X_φ = portion of X attributable to φ  (ANOVA decomposition)
    SVD(X_φ) → decoder D_φ (demixed components)
    regression  → encoder F_φ (biorthogonal to D_φ)
  → Z_φ = D_φᵀ X  (projected data for factor φ)

jPCA (applied to Z_φ):
  fit  Ż ≈ M_skew Z  where M_skew = −M_skewᵀ  (skew-symmetric)
  eigenvectors of M_skew → rotation planes
  → Z rotates in these planes over time
```

---

## 2. Overarching Architecture

```
EEG data (trial-averaged)
  X_avg : array [N_comp × T × n_rule × n_stim]
  X_trial: array [n_trials × N_comp × T × n_rule × n_stim]  ← optional, for regularization
  labels : "trs"   (t = time, r = rule, s = stim)

          │
          ▼
  ┌───────────────────┐
  │  dpca_marginalize │   ANOVA decomposition
  │  (X, labels)      │   X = X_t + X_r + X_s + X_tr + X_ts + X_rs + X_trs
  └───────────────────┘
          │
          ▼  named list: X_marginals[["t"]], [["r"]], [["s"]], ...
  ┌───────────────────┐
  │   dpca_fit        │   SVD per marginalization → D_φ (decoders)
  │   (X, labels,     │   regression              → F_φ (encoders)
  │    n_components)  │
  └───────────────────┘
          │
          ▼  model: list(D, F, labels, var_explained)
     ┌────┴────┐
     │         │
     ▼         ▼
dpca_transform  dpca_significance
(Z_φ = D_φᵀ X) (shuffle, optional)

     │
     ▼
  Z_r  [n_r_comp × T × n_rule]   ← rule subspace
  Z_s  [n_s_comp × T × n_stim]   ← stim subspace
  Z_t  [n_t_comp × T]            ← time subspace
  Z_rs [n_rs_comp × T × n_rule × n_stim]

     │
     ▼
  ┌───────────────────┐
  │   jpca_fit        │   fit M_skew per subspace
  │   (Z_list)        │   eigenvectors → jPC planes
  └───────────────────┘
          │
          ▼
  jpca_transform          project onto jPC1/jPC2
  jpca_rotation_strength  x→ẋ angle distribution (peak near π/2 = pure rotation)
```

**Four implementation phases:**

| Phase | Goal | Key functions |
|-------|------|---------------|
| **1** | Marginalization: ANOVA decomposition of X | `dpca_powersets`, `dpca_marginalize` |
| **2** | dPCA fit: SVD + encoders + significance | `dpca_fit`, `dpca_transform`, `dpca_significance` |
| **3** | jPCA: rotation detection per subspace | `jpca_fit`, `jpca_transform`, `jpca_rotation_strength` |
| **4** | Integration + EEG application | `dpca_jpca_pipeline`, apply to EEGMRI_RuleAction |

---

## 3. Input Data Format

### Raw h5 file structure (starting point)

```r
# h5read(f2load, "eegpower")  →  ds_eeg
#   4D (eegpower): [n_trial × n_time × n_freq × n_chan]
#   3D (eegraw):   [n_trial × n_time × n_chan]
#
# After dim adjustment in decoding script (line 138):
#   dim(d) <- c(dim(d)[1:2], prod(dim(d)[3:4]))
#   d: [n_trial × n_time × N]   N = n_freq * n_chan
#
# Trial metadata: ds_bl[[modelV]]  ← condition label per trial
# CV structure:   ds_bl[["balanceID"]]  ← balance fold ID (from preprop_decode)
#                 foldsL  ← CV fold list (from preprop_decode)
```

### Bridge function: `prep_cond_avg()` — already in `basic_lib.R`

The critical preparation step is already implemented in `basic_lib.R`:

```r
prep_cond_avg <- function(d, blIDX, grpIDX) {
  # [XnL] = prep_cond_avg(d, blIDX, grpIDX)
  # Condition-averaged data per balance fold → list of [N x T] matrices
  #
  # XnL   : list of [N x T] matrices, one per balance fold
  # d     : array [n_trial x n_time x N]
  # blIDX : balance fold ID per trial  (ds_bl[["balanceID"]])
  # grpIDX: condition label per trial  (ds_bl[[modelV]])
  Xn  <- narray::map(d, along=1, mean, subsets=paste0(blIDX,"_",grpIDX))
  Xn  <- Xn[order(rownames(Xn)),,]
  Xn  <- narray::split(Xn, along=1,
                       subsets=rep(unique(blIDX), each=dim(Xn)[1]/2))
  XnL <- lapply(Xn, function(x){t(dimAdj(x,c(1,2)))})
  return(XnL)
}
```

This is **directly extracted from EEGMRI_RuleAction_Decode_APPLY_mcca_.R lines 157–161**.
The `blIDX × grpIDX` grouping means CV fold and condition averaging are handled simultaneously —
the same CV structure (`foldsL`, `balanceID`) used for MCCA is reused for jPCA and dPCA.

### The "STEP 2 swap" pattern

MCCA, jPCA, and dPCA share identical data preparation. Only the algorithm in STEP 2 differs:

```r
foreach(f = names(foldsL)) %dopar% {
  trnIDX     <- foldsL[[f]]
  grpIDX_trn <- ds_bl[[modelV]][trnIDX]
  blIDX_trn  <- ds_bl[["balanceID"]][trnIDX]

  # ← prep_cond_avg: identical across all three methods
  XnL <- prep_cond_avg(d[trnIDX,,], blIDX_trn, grpIDX_trn)

  # ↓ STEP 2: swap algorithm here ↓ ──────────────────────

  # MCCA (current):
  Xn <- Reduce(cbind, XnL)
  g(A, score, AA) %=% nt_mcca(crossprod(Xn), ncol(XnL[[1]]))

  # jPCA (new):
  jpca_fit(XnL)

  # dPCA (new):
  dpca_fit(XnL, labels = "trs")

  # dPCA → jPCA (new):
  model_d <- dpca_fit(XnL, labels = "trs")
  jpca_fit(dpca_transform(XnL, model_d)[["r"]])  # rule subspace rotation

  # ───────────────────────────────────────────────────────
}
```

> **Comprehension Check:** `prep_cond_avg` returns a **list** of `[N × T]` matrices (one per balance fold).
> Why does jPCA want a list rather than a single `[N × T × n_cond]` array?
> Hint: Churchland 2012 subtracts the cross-condition mean before fitting M_skew.
> What does "condition" correspond to in the balance-fold structure?

### Note on weight-space jPCA (multi-class requirement)

```r
# Binary classification (e.g. RESP_ED: correct vs error):
#   LDA → 1 discriminant axis → projection is scalar → jPCA not meaningful
#
# Multi-class (e.g. PTSRCONJ: 12 conjunctions):
#   LDA → 11 discriminant axes → projection is 11-dimensional → jPCA applicable
#   XnL[[fold]][N_axes, T] is already the right shape for jpca_fit()
#
# Rule of thumb: n_conditions >= 3 for meaningful rotation planes.
# For binary outcomes, use raw EEG features (N = n_freq * n_chan) instead.
```

---

## 4. Mathematical Foundation: dPCA

> **Reference:** Kobak et al. (2016) *eLife* 5:e10989, Section "Methods: dPCA algorithm"

### Step 1: ANOVA-style marginalization

For data X[N, T, S, D] (neurons × time × stimulus × decision):

```
Grand mean:   μ[n]      = mean over T, S, D

Marginals (pure ANOVA effects, Yates decomposition):
  X_t[n,t]   = mean_{S,D}(X)[n,t]        - μ[n]
  X_s[n,s]   = mean_{T,D}(X)[n,s]        - μ[n]
  X_d[n,d]   = mean_{T,S}(X)[n,d]        - μ[n]
  X_ts[n,t,s]= mean_D(X)[n,t,s]          - X_t[n,t] - X_s[n,s] - μ[n]
  X_td[n,t,d]= mean_S(X)[n,t,d]          - X_t[n,t] - X_d[n,d] - μ[n]
  X_sd[n,s,d]= mean_T(X)[n,s,d]          - X_s[n,s] - X_d[n,d] - μ[n]
  X_tsd      = X - all lower-order terms

Verification: X - μ = X_t + X_s + X_d + X_ts + X_td + X_sd + X_tsd
              (these are orthogonal in the Frobenius sense)
```

Each X_φ is then **unfolded** to shape [N × K] where K = T × S × D.

> **Comprehension Check:** Why are the marginals computed as pure ANOVA effects (subtracting lower-order terms) rather than just conditional means?
> Hint: if you used `X_ts = mean_D(X)` directly, what variance would X_ts contain that dPCA does not want in the time×stimulus component?

### Step 2: Decoder computation (SVD)

For each marginalization φ:

```
SVD: X_φ[N × K] = U_φ Σ_φ Vᵀ_φ

D_φ = U_φ[:, 1:k_φ]   ← top k_φ left singular vectors
                         = the "demixed components" for factor φ
                         shape: [N × k_φ]

Explained variance: Σ_φ[i]² / ‖X_φ‖²_F
```

D_φ is analogous to PCA components, but computed only on variance attributable to φ.

### Step 3: Encoder computation (biorthogonal regression)

D_φ tells us how to *decode* (project data → low-dim). But for reconstruction, we need an *encoder* F_φ that maps back:

```
Z_φ = D_φᵀ X          (latent, k_φ × K)
F_φ = X_φ Zᵀ_φ (Z_φ Zᵀ_φ)⁻¹    (shape: N × k_φ)

Biorthogonality condition: D_φᵀ F_φ ≈ I_{k_φ}
→ ensures each latent dimension maps back to one task factor only
```

> **Comprehension Check:** What goes wrong if you skip the encoder and use D_φ for both decoding and encoding (i.e., set F_φ = D_φ as in standard PCA)?
> Hint: consider that D_φ is optimized for X_φ, but the projection Z_φ = D_φᵀ X captures variance from *all* marginalizations.

### Step 4: Regularization (optional)

When N > K (more channels than condition×time combinations), the regression for F_φ is ill-conditioned. Regularizer α is chosen by cross-validation on trial data X_trial:

```
F_φ = X_φ Zᵀ_φ (Z_φ Zᵀ_φ + α I)⁻¹

α = "auto": sweep over log-spaced values, select by held-out reconstruction loss
```

---

## 5. Mathematical Foundation: jPCA

> **Reference:** Churchland et al. (2012) *Nature* 487, 51–56, Supplementary Methods

### The rotation hypothesis

Given population state Z[k × T × C] (components × time × conditions), we ask whether dynamics are approximately rotational:

```
Ż(t) ≈ M_skew Z(t)

where M_skew = −M_skewᵀ   (skew-symmetric)

Why skew-symmetric?
  If M_skew is skew-symmetric → eigenvalues are purely imaginary: λ = ±iω
  Solution: Z(t) = e^{M_skew t} Z(0), and e^{M_skew} is orthogonal (rotation matrix)
  → population state rotates in the planes defined by eigenvectors of M_skew
```

### Algorithm

```
Input: Z_list  — list of condition matrices, each [k × T]
                 (k = n_components from dPCA or raw PCA; T = timepoints)

Step 1: PCA prefilter (if k > n_pcs)
  Stack all conditions: Z_full[k × C*T]
  Subtract cross-condition mean (per timepoint, averaged over conditions)
  PCA → keep top n_pcs PCs → X_red[n_pcs × C*T]

Step 2: Estimate M (unconstrained)
  Ẋ ≈ M X  →  M_hat = Ẋ Xᵀ (X Xᵀ)⁻¹
  where Ẋ[t] = X[t+1] − X[t]   (first difference approximation)

Step 3: Skew-symmetrize
  M_skew = (M_hat − M_hatᵀ) / 2

Step 4: Eigendecomposition of M_skew
  eigenvalues:  purely imaginary ±iω₁, ±iω₂, ...   (sorted by |ω| descending)
  eigenvectors: complex conjugate pairs (v₁, v̄₁), (v₂, v̄₂), ...

Step 5: Recover real rotation planes
  jPC1 = Re(v₁) + Re(v̄₁)  =  2 Re(v₁)
  jPC2 = Re(i(v₁ − v̄₁))   = −2 Im(v₁)
  (normalize to unit length; orient so net rotation is counter-clockwise)

Output: jPC axes W = [jPC1; jPC2]  (2 × n_pcs)
        projection X_jPCA = W X_red  (2 × C*T)
```

### Rotation strength metric

```
For each condition c and timepoint t:
  θ(t,c) = angle between x(t,c) and ẋ(t,c) in the jPCA plane
  θ = π/2  → pure rotation (ẋ perpendicular to x, i.e. tangent to circle)
  θ = 0    → pure expansion/contraction (ẋ parallel to x)

R²_skew / R²_unrestr:
  how well does M_skew fit vs unconstrained M?
  ratio close to 1 → dynamics are well-described by pure rotation
```

> **Comprehension Check:** Why is the cross-condition mean subtracted before jPCA (Step 1)?
> Hint: what kind of variance would dominate jPCA if a strong evoked response is present in all conditions?

---

## 6. Repository Structure

> **New repo, not a fork.** `AKikumoto/demixed_jPCA_R` is a standalone R implementation.
> `machenslab/dPCA` is cloned locally as `dPCA_reference/` (read-only numerical reference).

```
demixed_jPCA_R/              ← AKikumoto/demixed_jPCA_R  (new public repo)
├── R/
│   ├── jpca_lib.R           # jPCA functions (jpca_* prefix) — implement first
│   ├── dpca_lib.R           # dPCA functions (dpca_* prefix)
│   └── dpca_jpca_pipeline.R # integrated pipeline
├── tests/
│   ├── test_jpca_fit.R      # Phase 0: jPCA numerical tests
│   ├── test_marginalize.R   # Phase 1: ANOVA decomposition tests
│   ├── test_dpca_fit.R      # Phase 2: dPCA fit/transform tests
│   └── test_pipeline.R      # Phase 3: end-to-end test
├── notebooks/
│   ├── 01_jpca_tutorial.Rmd         # Churchland 2012 Fig.3 reproduction
│   ├── 02_dpca_tutorial.Rmd         # Kobak 2016 Fig.2 reproduction
│   └── 03_eeg_application.Rmd       # EEGMRI_RuleAction full pipeline
├── data/reference/
│   ├── jpca_matlab_output.rds        # Churchland MATLAB output for verification
│   └── dpca_python_output.rds        # machenslab/dPCA Python output for verification
├── manuscripts/
│   └── ARCHITECTURE_demixed_jPCA.md  # ← this file
└── README.md                          # credits machenslab/dPCA and Churchland lab

dPCA_reference/              ← git clone machenslab/dPCA  (read-only, local only)
```

**Dependency on `basic_lib.R`:**
`prep_cond_avg()` lives in `basic_lib.R` and is the only EEG-specific prep function.
`jpca_lib.R` and `dpca_lib.R` have no dependency on `basic_lib.R` — they are data-source agnostic.

---

## 7. `R/dpca_lib.R` — dPCA Functions

> **Note on data preparation:** `prep_cond_avg()` (in `basic_lib.R`) handles all
> EEG data preparation for both jPCA and dPCA. No prep function lives in `dpca_lib.R`.
> All `dpca_*` functions receive `XnL` (list of `[N × T]` matrices) directly.

### `dpca_powersets(labels)`

```r
dpca_powersets <- function(labels) {
  # Generate all non-empty subsets of parameter labels
  #
  # Returns: character vector of all subsets, sorted by size then alphabetically
  #          e.g. "trs" → c("r","s","t","rs","rt","st","rst")
  #
  # labels: character string of parameter labels, e.g. "trs"
  #         convention: 't' = time (first character)
  # --------------------------------------------------------
}
```

> **Comprehension Check:** For labels = "trs", how many marginalizations does dPCA compute?
> Which subset corresponds to the "condition-independent dynamics" component?

### `dpca_marginalize(X, labels)`

```r
dpca_marginalize <- function(X, labels) {
  # Compute ANOVA-style marginalized data matrices (Yates decomposition)
  #
  # Returns: named list of matrices, each [N × K] where K = prod(dim(X)[-1])
  #          names match subsets from dpca_powersets(labels)
  #          e.g. list("t" = matrix[N,K], "r" = matrix[N,K], ...)
  #
  # X:      array [N × d1 × d2 × ... × dk]
  #         N = neurons/components, d1..dk correspond to labels
  # labels: character string of length k, e.g. "trs"
  #
  # Algorithm:
  #   1. grand_mean <- apply(X_2d, 1, mean)  → [N]
  #   2. For each subset phi (by size, smallest first):
  #      a. avg_dims  <- dimensions NOT in phi (excl. neuron dim)
  #      b. X_phi_cm  <- conditional mean: average over avg_dims, broadcast to dim(X)
  #      c. X_phi     <- X_phi_cm - grand_mean_broadcast
  #                      - sum of all pure marginals of strict subsets of phi
  #      (inclusion-exclusion / Möbius inversion on subset lattice)
  #   3. Unfold each X_phi to [N × K]
  #
  # Verification: sum of all marginals == X_2d - grand_mean_broadcast (to machine precision)
  # --------------------------------------------------------
  if (length(dim(X)) != nchar(labels) + 1) {
    stop("dim(X) must be [N, d1, ..., dk] where k = nchar(labels)")
  }
}
```

> **Comprehension Check:** Write out by hand the marginalization formulas for a toy array X[2, 3, 2] with labels = "ts".
> Verify that X_t + X_s + X_ts == X - grand_mean when X is [2 × 3 × 2].
> Do this *before* implementing the function.

### `dpca_fit(X, labels, n_components, regularizer)`

```r
dpca_fit <- function(X, labels, n_components = 3, regularizer = NULL, X_trial = NULL) {
  # Fit dPCA model: compute decoders D and encoders F for each marginalization
  #
  # Returns: list with elements
  #   $D         : named list of decoder matrices, D[[phi]] is [N × k_phi]
  #   $F         : named list of encoder matrices, F[[phi]] is [N × k_phi]
  #   $var_exp   : named list of explained variance ratios per marginalization
  #   $labels    : labels string (stored for dpca_transform)
  #   $dim_X     : dim(X) (stored for reshaping in dpca_transform)
  #   $alpha     : regularizer value used (NULL if none)
  #
  # X:            array [N × T × n_param1 × ...]
  # labels:       "trs" etc.
  # n_components: integer (same for all phi) OR named list e.g. list(t=3, r=3, s=3, rs=2)
  # regularizer:  NULL | numeric value | "auto" (cross-validated from X_trial)
  # X_trial:      array [n_trial × N × T × ...] optional; required if regularizer="auto"
  #
  # Algorithm:
  #   1. X_marginals <- dpca_marginalize(X, labels)
  #   2. X_2d        <- matrix(X, nrow=N, ncol=K)      ← unfold full data
  #   3. For each phi:
  #      a. g(U, s, V) %=% svd(X_marginals[[phi]])
  #      b. D[[phi]] <- U[, 1:k_phi]
  #      c. Z_phi    <- t(D[[phi]]) %*% X_2d           ← project full data
  #      d. F[[phi]] <- X_marginals[[phi]] %*% t(Z_phi) %*%
  #                     solve(Z_phi %*% t(Z_phi) + alpha * diag(k_phi))
  #      e. var_exp[[phi]] <- cumsum(s^2) / sum(s^2)
  # --------------------------------------------------------
}
```

### `dpca_transform(X, model)`

```r
dpca_transform <- function(X, model) {
  # Project data onto dPCA components
  #
  # Returns: named list of projected arrays
  #   Z[[phi]] : array [k_phi × T × n_param1 × ...]
  #              (reshaped from [k_phi × K])
  #
  # X:     array [N × T × ...] (same layout as in dpca_fit)
  # model: output of dpca_fit
  #
  # Z[[phi]] = D[[phi]]^T %*% X_2d  then reshape to [k_phi, T, ...]
  # --------------------------------------------------------
}
```

### `dpca_significance(model, X, X_trial, n_splits, n_shuffles)`

```r
dpca_significance <- function(model, X, X_trial, n_splits = 10, n_shuffles = 100) {
  # Cross-validated significance of dPCA components via label shuffling
  #
  # Returns: named list of logical matrices, sig[[phi]] is [k_phi × 1]
  #          TRUE = component is significant at p < 0.05
  #
  # Method (Kobak et al. 2016, Methods):
  #   Split X_trial into train/test halves n_splits times.
  #   Fit dPCA on train; project test onto D[[phi]].
  #   Classification score = fraction correct in 1-NN classifier on Z[[phi]].
  #   Compare to null: shuffle condition labels n_shuffles times.
  #   Component significant if true score > max(null scores).
  # --------------------------------------------------------
}
```

---

## 8. `R/jpca_lib.R` — jPCA Functions

### `jpca_fit(X_list, n_pcs)`

```r
jpca_fit <- function(X_list, n_pcs = 6) {
  # Fit jPCA model: find rotation planes in population dynamics
  #
  # Returns: list with elements
  #   $W         : jPC axes matrix [2 × n_pcs] (first rotation plane)
  #   $W_all     : all jPC pairs [n_pcs × n_pcs]
  #   $M_skew    : fitted skew-symmetric matrix [n_pcs × n_pcs]
  #   $M_unrestr : unconstrained M [n_pcs × n_pcs]
  #   $R2_skew   : fit quality of M_skew
  #   $R2_unrestr: fit quality of unconstrained M
  #   $pca       : prcomp result (for projecting new data)
  #   $eig_freq  : rotation frequencies |Im(eigenvalues)|, sorted descending
  #
  # X_list: list of matrices, each [N × T]
  #         one matrix per condition; N = components, T = timepoints
  #         (output of dpca_transform for one marginalization)
  #
  # Algorithm:
  #   1. Stack: X_full[N × C*T] <- do.call(cbind, X_list)
  #   2. Subtract cross-condition mean per timepoint (soft-normalization optional)
  #   3. PCA: pca <- prcomp(t(X_full), center=FALSE); X_red <- t(pca$rotation[,1:n_pcs]) %*% X_full
  #   4. Finite difference: dX <- X_red[, 2:end] - X_red[, 1:(end-1)]
  #                         X_prev <- X_red[, 1:(end-1)]
  #   5. Fit unconstrained M: M_hat <- dX %*% t(X_prev) %*% solve(X_prev %*% t(X_prev))
  #   6. Skew-symmetrize: M_skew <- (M_hat - t(M_hat)) / 2
  #   7. Eigen: g(values, vectors) %=% eigen(M_skew)
  #   8. Sort by |Im(values)| descending; recover real planes from complex pairs
  #   9. Orient jPC1/jPC2 so mean rotation is counter-clockwise
  # --------------------------------------------------------
  if (!is.list(X_list)) stop("X_list must be a list of matrices (one per condition)")
}
```

### `jpca_transform(X_list, model)`

```r
jpca_transform <- function(X_list, model) {
  # Project data onto jPCA planes
  #
  # Returns: list with elements
  #   $proj      : matrix [2 × C*T] (projection onto first jPCA plane)
  #   $proj_list : list of [2 × T] matrices, one per condition
  #
  # X_list: list of condition matrices (same format as jpca_fit input)
  # model:  output of jpca_fit
  # --------------------------------------------------------
}
```

### `jpca_rotation_strength(proj_list, X_list)`

```r
jpca_rotation_strength <- function(proj_list, model) {
  # Compute rotation strength: distribution of angle between x and ẋ in jPCA plane
  #
  # Returns: list with elements
  #   $angles    : vector of angles θ (one per time×condition; peak near π/2 = rotation)
  #   $peak      : mode of angle distribution
  #   $R2_ratio  : R²_skew / R²_unrestr  (1.0 = perfect rotation)
  #
  # Method (Churchland et al. 2012, Fig. 6a):
  #   For each (t, c): θ(t,c) = atan2(x ∧ ẋ, x · ẋ)
  #   where ∧ is the 2D cross product (signed area) and · is dot product
  # --------------------------------------------------------
}
```

---

## 9. `R/dpca_jpca_pipeline.R` — Integrated Pipeline

```r
dpca_jpca_pipeline <- function(X_avg, labels, 
                                n_dpca_components = 3, 
                                n_jpca_pcs = 6,
                                regularizer = NULL,
                                X_trial = NULL) {
  # Full dPCA → jPCA pipeline
  #
  # Returns: list with elements
  #   $dpca_model   : output of dpca_fit
  #   $Z            : output of dpca_transform (named list of projected arrays)
  #   $jpca_models  : named list of jpca_fit outputs, one per marginalization
  #   $projections  : named list of jpca_transform outputs
  #   $rot_strength : named list of jpca_rotation_strength outputs
  #
  # Usage:
  #   result <- dpca_jpca_pipeline(X_avg, labels = "trs",
  #                                n_dpca_components = list(t=3, r=3, s=3, rs=2))
  #   # Visualize rule subspace rotation:
  #   plot_jpca_plane(result$projections[["r"]], conditions = rule_labels)
  # --------------------------------------------------------
  dpca_model <- dpca_fit(X_avg, labels, n_components = n_dpca_components,
                         regularizer = regularizer, X_trial = X_trial)
  Z <- dpca_transform(X_avg, dpca_model)
  
  jpca_models  <- list()
  projections  <- list()
  rot_strength <- list()
  
  for (phi in names(Z)) {
    # Convert Z[[phi]] to list of condition matrices for jpca_fit
    X_list_phi <- dpca_to_jpca_input(Z[[phi]], phi, labels)
    jpca_models[[phi]]  <- jpca_fit(X_list_phi, n_pcs = n_jpca_pcs)
    projections[[phi]]  <- jpca_transform(X_list_phi, jpca_models[[phi]])
    rot_strength[[phi]] <- jpca_rotation_strength(projections[[phi]], jpca_models[[phi]])
  }
  
  list(dpca_model = dpca_model, Z = Z,
       jpca_models = jpca_models, projections = projections,
       rot_strength = rot_strength)
}
```

---

## 10. Test Strategy

### Numerical verification against Python reference

Every function must be verified against `machenslab/dPCA` (Python) on identical toy data:

```r
# tests/test_dpca_fit.R
# ---------------------------------------------------------
# 1. Generate toy data in R: set.seed(42); X <- array(rnorm(2*5*3*2), dim=c(2,5,3,2))
# 2. Save to Python: write.csv(as.data.frame(X), "toy_X.csv")
# 3. Run Python dPCA on toy_X.csv → save D, F, Z as CSVs
# 4. Load Python output into R: D_py <- read.csv("D_phi.csv")
# 5. Assert: max(abs(D_r - D_py)) < 1e-6  (up to sign flip per column)
# ---------------------------------------------------------

test_dpca_marginalize_sums_to_X <- function() {
  # Verify: sum of all marginals == X - grand_mean  (Frobenius norm of residual < 1e-10)
}

test_dpca_biorthogonality <- function() {
  # Verify: t(D[[phi]]) %*% F[[phi]] ≈ diag(k_phi)  for all phi
}

test_jpca_skewsymmetric <- function() {
  # Verify: M_skew + t(M_skew) ≈ 0  (entries < 1e-10)
}

test_jpca_eigenvalues_imaginary <- function() {
  # Verify: all Re(eigenvalues(M_skew)) ≈ 0
}

test_rotation_strength_surrogate <- function() {
  # Generate data with known pure rotation (Z(t) = R(ωt) Z(0))
  # Verify: peak angle ≈ π/2,  R2_ratio ≈ 1.0
}
```

---

## 11. Implementation Order

> **Learning principle: jPCA first, then dPCA.**
> jPCA has fewer moving parts and can be validated immediately on real EEG data.
> dPCA builds on the same geometric intuition but adds ANOVA decomposition complexity.
> Understand rotation before understanding demixing.

```
─────── Phase 0: Standalone jPCA (start here) ───────
Step 1   R/jpca_lib.R: jpca_fit()
         tests/test_jpca_fit.R:
           - M_skew + t(M_skew) ≈ 0
           - eigenvalues purely imaginary
           - jPC1 ⊥ jPC2
           - match MATLAB output on Churchland public data (Monkey N)

Step 2   R/jpca_lib.R: jpca_transform(), jpca_rotation_strength()
         tests/test_jpca_fit.R:
           - surrogate rotation data: peak angle ≈ π/2, R2_ratio ≈ 1.0
           - random data: R2_ratio ≈ 0.5

Step 3   Apply to EEGMRI_RuleAction (PTSRCONJ, ≥3 conditions):
         Plug jpca_fit(XnL) into MCCA script STEP 2 swap pattern
         → First real rotation result before any dPCA code exists

─────── Phase 1: dPCA Marginalization ───────
Step 4   R/dpca_lib.R: dpca_powersets(labels)
         tests/test_marginalize.R: verify subset count, sorting

Step 5   R/dpca_lib.R: dpca_marginalize(XnL, labels)
         Hand-trace on toy XnL (3 conditions, 5 timepoints, N=2) before coding
         tests/test_marginalize.R:
           - sum of marginals == X - grand_mean
           - orthogonality: Frobenius inner product of different marginals ≈ 0
           - match Python output on same toy data

─────── Phase 2: dPCA fit ───────
Step 6   R/dpca_lib.R: dpca_fit() — decoder only (SVD step)
Step 7   R/dpca_lib.R: dpca_fit() — add encoder (regression step)
         tests/test_dpca_fit.R:
           - biorthogonality: t(D[[phi]]) %*% F[[phi]] ≈ I
           - match Python D, F matrices (up to sign flip)

Step 8   R/dpca_lib.R: dpca_transform(XnL, model)
         tests/test_dpca_fit.R: compare Z[[phi]] to Python output

Step 9   dpca_significance() (optional)

─────── Phase 3: dPCA → jPCA integration ───────
Step 10  R/dpca_jpca_pipeline.R: dpca_jpca_pipeline()
         tests/test_pipeline.R: end-to-end on surrogate data

Step 11  Apply to EEGMRI_RuleAction:
         Primary question: does rule subspace (Z_r) show stronger rotation
         than stimulus subspace (Z_s)?
         Compare R2_ratio: jpca_fit(Z_r) vs jpca_fit(Z_s)

─────── Phase 4: Notebooks ───────
Step 12  notebooks/02_dpca_tutorial.Rmd  — reproduce Kobak 2016 Fig.2 on surrogate data
Step 13  notebooks/03_jpca_tutorial.Rmd  — reproduce Churchland 2012 Fig.3 on public data
Step 14  notebooks/04_eeg_application.Rmd — full pipeline on EEGMRI_RuleAction
```

---

## 12. Coding Conventions

All functions follow the `noiseTools_lib.R` style established in the MCCA conversion:

```r
# 1. Multiple assignment operator (already defined in noiseTools_lib.R)
#    g(a, b) %=% list(val1, val2)

# 2. Function documentation format
my_func <- function(x, param = NULL) {
  # [output1, output2] = my_func(x, param) — one-line description
  #
  # output1: description and shape
  # output2: description and shape
  #
  # x:     input description and shape
  # param: optional parameter description
  # --------------------------------------------------------
  if (is.null(x)) stop("x must be provided")
  # ... implementation
  return(list(output1 = ..., output2 = ...))
}

# 3. SVD convention: g(U, s, Vt) %=% svd(M) returns d=s, u=U, v=V (not Vt!)
#    → M ≈ U %*% diag(s) %*% t(V)   [R convention, different from numpy]

# 4. Array dimension order: always [N_neurons × ...condition_dims]
#    N is always dim 1 (neurons/components) — never collapse it

# 5. Tests: every function has a corresponding test_*.R file.
#    Tests run with source("tests/test_*.R") and print PASS/FAIL explicitly.
```

---

## 13. Design Principles

1. **Understanding before code.** Every function maps to specific equations in Kobak et al. (2016) or Churchland et al. (2012). Write the equation number in the comment before the corresponding line.

2. **Test against Python/MATLAB reference on identical data.** Numerical agreement (max absolute error < 1e-6) is required before any function is considered complete. Sign flips in eigenvectors are acceptable and must be handled in the comparison.

3. **`prep_cond_avg()` is the only EEG-specific prep.** It lives in `basic_lib.R`, not in `dpca_lib.R` or `jpca_lib.R`. All `dpca_*` and `jpca_*` functions receive `XnL` (list of `[N × T]` matrices) directly and are data-source agnostic.

4. **dPCA and jPCA are independent modules.** `jpca_lib.R` has no dependency on `dpca_lib.R`. jPCA can be applied to raw EEG, MCCA-denoised data, or dPCA-projected subspaces equally.

5. **CV structure is shared with MCCA.** Use the same `foldsL` and `balanceID` from `preprop_decode()`. No new CV infrastructure is needed for jPCA or dPCA.

6. **jPCA requires multi-class conditions.** Binary outcomes (e.g. RESP_ED) produce a 1D projection — jPCA is not meaningful. Use conditions with ≥ 3 levels (e.g. PTSRCONJ) or raw EEG features.

7. **TDR is not the right approach for categorical variables.** TDR (regression-based) is optimal for continuous task variables (as in Mante et al. 2013). For categorical EEG factors (rule, stim, resp), the dummy-coded axes are arbitrary. dPCA's SVD-based approach is coding-independent and is the correct method.

8. **Repository: new repo, not a fork.** `AKikumoto/demixed_jPCA_R` is a standalone R implementation. `machenslab/dPCA` is cloned locally as a read-only numerical reference only. README credits the original.

9. **Marginalization is the conceptual core of dPCA.** If `dpca_marginalize` is wrong, everything else is wrong. Hand-trace on toy data (3 conditions × 5 timepoints × N=2) before coding.

10. **jPCA first, dPCA second.** jPCA can be validated on real data immediately (Phase 0). dPCA depends on understanding the geometric intuition jPCA provides.

---

## 14. Connection to Existing Projects

| Analysis | demixed_jPCA equivalent |
|----------|------------------------|
| Single-trial LDA decoding (EEGMRI_RuleAction) | N = LDA discriminant axes → X_avg[N, T, n_rule, n_stim] → dpca_fit() |
| RSA with temporal generalization (K&M 2020 style) | Compare to Z[[phi]] decoding over time (temporal dPCA decoding curves) |
| EmbeddingRNN conjunctive gradient steepness | dPCA rule subspace variance ratio: high conjunction necessity → higher Var(Z_r)/Var(Z_s) |
| RotationRNN jPCA on h_t | Same `jpca_lib.R` functions; apply to RNN hidden states as X_list |
| EmbeddingRNN `units_to_rsa()` | dpca_transform() provides Z_phi; RSA within each subspace |
| BG-ACC step-by-step implementation discipline | Same approach: implement each function once, test immediately, no refactoring mid-way |

---

## 15. Open Questions

- **Soft normalization in jPCA:** Churchland applies per-neuron range normalization before jPCA. Is this appropriate when N = LDA components (already normalized)? → Default off; expose as optional parameter.
- **Cross-condition mean subtraction in jPCA:** Required for raw neurons; may remove meaningful temporal structure in dPCA-projected Z_r. → Test both on EEGMRI_RuleAction data.
- **n_components selection for dPCA:** how many components per marginalization for EEG data with N ≈ 20–50? → Plot scree per marginalization; use elbow or 90% variance cutoff.
- **jPCA on dPCA subspace vs raw EEG:** which gives stronger rotation signal for rule subspace? → Compare in Step 14 notebook.
- **Connection to Naomi Feldman collaboration:** dPCA on speech representation data (neural geometry of phonological distinctions) uses the same pipeline with different label structure.

---

## 16. References

- **Kobak, D. et al. (2016).** Demixed principal component analysis of neural population data. *eLife* 5:e10989. ← dPCA algorithm, full mathematical derivation
- **Churchland, M.M. et al. (2012).** Neural population dynamics during reaching. *Nature* 487, 51–56. ← jPCA, rotational dynamics, rotation strength metric
- **Kikumoto, A. et al. (2024).** Conjunctive representations that combine stimuli, responses, and rules are critical for action selection. *Nature Communications* ← experimental context, EEG decoding geometry
- **Kikumoto, A. & Mayr, U. (2020).** Conjunctive representations that integrate stimuli, responses, and rules are critical for action selection. *PNAS* 117(19), 10603–10608. ← RSA framework for EEG decoding
- **Brendel, W. et al. (2011).** Demixed principal component analysis. *NIPS 2011.* ← original dPCA paper (predecessor to Kobak 2016)
- **machenslab/dPCA** (Python/MATLAB). https://github.com/machenslab/dPCA ← reference implementation for numerical verification
- **Churchland lab code** (MATLAB jPCA). https://churchland.zuckermaninstitute.columbia.edu/content/code ← reference for jPCA numerical verification
- **noiseTools_lib.R** (this project). MCCA conversion from MATLAB → R. ← coding convention reference
