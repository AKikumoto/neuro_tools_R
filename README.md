# neuro\_tools\_R

R implementations of neural data analysis methods, translated from published MATLAB/Python originals.

| Method | Reference |
|--------|-----------|
| dPCA | Kobak et al. (2016) *eLife* 5:e10989 |
| jPCA | Churchland et al. (2012) *Nature* 487, 51–56 |
| RRR | Wu & Pillow (2025) *arXiv* 2512.12467 |

---

## Original Work & Attribution

All scientific credit belongs to the original authors.
These R libraries are independent re-implementations written from scratch using published algorithms as reference.
They do not wrap, link to, or redistribute the original code.

### demixed PCA (dPCA)

**Paper:**
> D Kobak, W Brendel, C Constantinidis, CE Feierstein, A Kepecs, ZF Mainen,
> X-L Qi, R Romo, N Uchida, CK Machens
> *Demixed principal component analysis of neural population data*
> **eLife** 2016, 5:e10989. DOI: 10.7554/eLife.10989

**Original code (Python + MATLAB):** <https://github.com/machenslab/dPCA>

### jPCA

**Paper:**
> MM Churchland, J Cunningham, MT Kaufman, JD Foster, P Nuyujukian, SI Ryu, KV Shenoy
> *Neural population dynamics during reaching*
> **Nature** 2012, 487, 51–56. DOI: 10.1038/nature11129

**Original code (MATLAB):** Shenoy Lab, Stanford.

### Reduced Rank Regression (RRR)

**Paper:**
> B Wu & JW Pillow
> *Reduced rank regression for neural communication: a tutorial for neuroscientists*
> **arXiv** 2025, 2512.12467

**Original code (Python + MATLAB):** <https://github.com/bichanw/RRR>

---

## Overview

Each library implements one published method in R, following the math in the original paper equation by equation.

**dPCA** (`demixedPCA_lib.R`) — ANOVA-style decomposition of population activity into parameter-specific subspaces (time, stimulus, rule, interactions).

**jPCA** (`jPCA_lib.R`) — detects rotational dynamics within a subspace by fitting a skew-symmetric dynamics matrix M_skew to population trajectories.

**RRR** (`RRR_lib.R`) — finds the low-dimensional communication subspace between two neural populations (input region X → output region Y) via rank-constrained regression.

---

## Repository Structure

```
neuro_tools_R/
├── demixedPCA_lib.R        # dPCA implementation
├── jPCA_lib.R              # jPCA implementation
├── RRR_lib.R               # RRR implementation
│
├── notebook/
│   ├── test_demixedPCA.R   # dPCA unit tests (testthat)
│   ├── test_jpca_fit.R     # jPCA unit tests (testthat)
│   └── test_RRR.R          # RRR unit tests (testthat)
│
├── resource/
│   ├── ARCHITECTURE_demixed_j_PCA.md   # dPCA + jPCA design document
│   ├── ARCHITECTURE_RRR.md             # RRR design document
│   ├── jPCA_notes_en.md
│   ├── demixedPCA_notes_en.md
│   ├── jPCA_test_results_en.md
│   ├── demixedPCA_test_results_en.md
│   ├── jPCA_geometry.html
│   └── dPCA_anova_decomp.html
│
├── manuscripts/            # source papers (gitignored)
├── original/               # reference implementations (gitignored)
└── visualizations/
    ├── vis_jpca_test.R
    └── *.png
```

---

## Functions

### `demixedPCA_lib.R`

| Function | Description |
|---|---|
| `dpca_get_marginalizations(labels)` | Enumerate all 2^K − 1 parameter subsets |
| `dpca_marginalize(X, labels)` | ANOVA decomposition → orthogonal marginals |
| `dpca_fit(X, labels, n_components, regularizer)` | Fit dPCA; returns encoders `P` and decoders `D` |
| `dpca_transform(X, model)` | Project data; returns named list of arrays |
| `dpca_inverse_transform(Z, model, marginalization)` | Reconstruct one marginalization |
| `dpca_reconstruct(X, model, marginalization)` | Fit-transform-inverse in one call |
| `dpca_significance(...)` | Shuffle-test significance per component and timepoint |
| `dpca_plot(Z, model, ...)` | Full-figure plot (MATLAB dpca_plot equivalent) |

Input: `X` — array `[N × d1 × ... × dK]`, trial-averaged, N = neurons/channels.

### `jPCA_lib.R`

| Function | Description |
|---|---|
| `jpca_fit(X_list, n_pcs, normalize)` | Fit jPCA; returns `W`, `M_skew`, `R2_skew`, `eig_freq` |
| `jpca_transform(X_list, model)` | Project onto jPC plane; returns `proj` and `proj_list` |
| `jpca_rotation_strength(proj, model)` | Per-timepoint angles θ, peak, and R2_ratio |

Input: `X_list` — list of `[N × T]` matrices, one per condition (min 3 conditions).

### `RRR_lib.R`

| Function | Description |
|---|---|
| `rrr_simulate(T, nx, ny, rank, sigma_noise, Sigma)` | Generate data from RRR generative model |
| `rrr_fit(X, Y, rank, lambda)` | Fit RRR with optional ridge; returns `U`, `V`, `W` |
| `rrr_transform(X, model)` | Predict Y_hat = X W |
| `rrr_r2(X, Y, model)` | R² on training or held-out data |
| `rrr_cv_rank(X, Y, max_rank, lambda_grid, n_folds)` | Cross-validated rank and lambda selection |
| `rrr_fit_noniso(X, Y, rank, max_iter, tol)` | Full-covariance RRR (non-spherical noise) |
| `rrr_alignment_input(X, W, r, C)` | Input alignment index alpha_in ∈ [0, 1] |
| `rrr_alignment_output(X, Y, W)` | Output alignment index alpha_out ∈ [0, 1] + comm_frac |

Input: `X` — matrix `[T × m]`, `Y` — matrix `[T × n]`, both centered per column.

---

## Dependencies

```r
install.packages(c("MASS", "testthat", "ggplot2", "patchwork"))
```

Optional for Python comparison tests in `notebook/test_RRR.R`:

```r
install.packages("reticulate")
# requires numpy and scipy in the Python environment
```

---

## Mathematical Background

- [resource/ARCHITECTURE_demixed_j_PCA.md](resource/ARCHITECTURE_demixed_j_PCA.md) — dPCA + jPCA full design document
- [resource/ARCHITECTURE_RRR.md](resource/ARCHITECTURE_RRR.md) — RRR full design document
- [resource/jPCA_notes_en.md](resource/jPCA_notes_en.md) — skew-symmetric dynamics, θ geometry, R² ratio
- [resource/demixedPCA_notes_en.md](resource/demixedPCA_notes_en.md) — ANOVA decomposition, encoder/decoder asymmetry
