# demixed\_jPCA (R)

R implementations of **demixed PCA (dPCA)** and **jPCA** for analysing population-level neural / EEG dynamics.

| Method | Reference |
|--------|-----------|
| dPCA | Kobak et al. (2016) *eLife* 5:e10989 |
| jPCA | Churchland et al. (2012) *Nature* 487, 51–56 |

---

## Original Work & Attribution

This repository implements two published methods in R.
All scientific credit belongs to the original authors.

### demixed PCA (dPCA)

**Paper:**
> D Kobak, W Brendel, C Constantinidis, CE Feierstein, A Kepecs, ZF Mainen,
> X-L Qi, R Romo, N Uchida, CK Machens  
> *Demixed principal component analysis of neural population data*  
> **eLife** 2016, 5:e10989  
> <https://elifesciences.org/content/5/e10989>  
> DOI: 10.7554/eLife.10989

**Original code (Python + MATLAB):**  
<https://github.com/machenslab/dPCA>  
Maintained by the Machens Lab (BCCN Munich / Champalimaud Centre for the Unknown).  
The reference implementation (Python `dPCA` class and MATLAB `dpca_*.m` functions)
is included in `dPCA/` for reference.

### jPCA

**Paper:**
> MM Churchland, J Cunningham, MT Kaufman, JD Foster, P Nuyujukian,
> SI Ryu, KV Shenoy  
> *Neural population dynamics during reaching*  
> **Nature** 2012, 487, 51–56  
> DOI: 10.1038/nature11129

**Original code (MATLAB):**  
Distributed by the Shenoy Lab (Stanford).  
The original MATLAB scripts are the basis for the `jPCA_lib.R` implementation.

### Relationship to original code

Both R libraries are independent re-implementations written from scratch, using only
the published algorithms as reference.  They do **not** wrap, link to, or redistribute
the original code.  The `dPCA/` subdirectory contains the upstream Python package
as a read-only reference copy (original license: `dPCA/License.md`).

---

## Overview

Standard PCA mixes task-parameter variance (stimulus, rule, time) into each component, making interpretation difficult.
This project builds two complementary analyses:

1. **dPCA** (`demixedPCA_lib.R`) — ANOVA-style decomposition of population activity into parameter-specific subspaces.
2. **jPCA** (`jPCA_lib.R`) — detects rotational dynamics within a subspace by fitting a skew-symmetric dynamics matrix.

The intended pipeline is:

```
raw EEG [N × T × conditions]
    ↓  dPCA
rule subspace Z_r,  stimulus subspace Z_s,  interaction Z_rs, ...
    ↓  jPCA (per subspace)
rotation strength R²_ratio,  peak angle θ,  jPC plane visualisation
```

---

## Repository Structure

```
demixed_j_PCA/
├── jPCA_lib.R              # jPCA implementation
├── demixedPCA_lib.R        # dPCA implementation
│
├── test/
│   ├── test_jpca_fit.R     # 10 unit tests (testthat) — all PASS
│   └── test_demixedPCA.R   # 15 unit tests (testthat) — all PASS
│
├── resource/
│   ├── jPCA_notes_en.md          # jPCA math notes (English)
│   ├── demixedPCA_notes_en.md    # dPCA math notes (English)
│   ├── jPCA_test_results_en.md   # test explanation (English)
│   ├── demixedPCA_test_results_en.md
│   ├── jPCA_geometry.html        # visual supplement for jPCA geometry
│   └── dPCA_anova_decomp.html    # visual supplement for ANOVA decomposition
│
├── ref/
│   └── demixedPCA_notes2.md      # English math notes (reference)
│
├── visualizations/
│   ├── vis_jpca_test.R           # ggplot2 figures
│   ├── fig1_trajectories.png
│   ├── fig2_angle_distributions.png
│   └── fig3_R2_comparison.png
│
└── original/                     # reference Python / MATLAB implementation
    └── dPCA/
```

---

## Functions

### `jPCA_lib.R`

| Function | Description |
|---|---|
| `jpca_fit(X_list, n_pcs, normalize)` | Fit jPCA model; returns `W`, `M_skew`, `R2_skew`, `R2_unrestr`, `eig_freq` |
| `jpca_transform(X_list, model)` | Project data onto jPC plane; returns `proj` [2 × C×T] and `proj_list` |
| `jpca_rotation_strength(proj, model)` | Compute per-timepoint angles θ, peak, and `R2_ratio` |

**Input format**: `X_list` — a list of `[N × T]` matrices, one per condition (min 3 conditions).

### `demixedPCA_lib.R`

| Function | Description |
|---|---|
| `dpca_get_marginalizations(labels)` | Enumerate all 2^K − 1 parameter subsets |
| `dpca_marginalize(X, labels)` | ANOVA decomposition → orthogonal marginals |
| `dpca_fit(X, labels, n_components, regularizer)` | Fit dPCA; returns `P` (encoders), `D` (decoders) |
| `dpca_transform(X, model)` | Project data; returns named list of arrays with `explained_variance_ratio` attribute |
| `dpca_inverse_transform(Z, model, marginalization)` | Reconstruct single marginalization |
| `dpca_reconstruct(X, model, marginalization)` | Full reconstruction pipeline |
| `dpca_significance(...)` | Shuffle-test significance per component and time point |

**Input format**: `X` — array `[N × d1 × ... × dK]`, trial-averaged.

---

## Running Tests

```r
# jPCA
Rscript -e "library(testthat); source('jPCA_lib.R'); source('test/test_jpca_fit.R')"

# dPCA — requires a runner that sources demixedPCA_lib.R first
Rscript /tmp/run_dpca_tests.R
```

---

## Dependencies

```r
install.packages(c("MASS", "testthat", "ggplot2"))
```

---

## Mathematical Background

- [resource/jPCA_notes_en.md](resource/jPCA_notes_en.md) — skew-symmetric dynamics, θ geometry, R² ratio
- [resource/demixedPCA_notes_en.md](resource/demixedPCA_notes_en.md) — ANOVA decomposition, closed-form solution, encoder/decoder asymmetry
- [resource/jPCA_geometry.html](resource/jPCA_geometry.html) — visual supplement (jPC plane, θ cases)
- [resource/dPCA_anova_decomp.html](resource/dPCA_anova_decomp.html) — visual supplement (ANOVA step-by-step)
