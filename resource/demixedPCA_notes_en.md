# Demixed PCA — Mathematical Notes

**Reference**: Kobak D, Brendel W, et al. (2016). *Demixed principal component analysis of neural population data.* eLife 5:e10989.

**Python implementation**: https://github.com/machenslab/dPCA\
**R implementation**: `demixedPCA_lib.R` (this project)

------------------------------------------------------------------------

## 1. Motivation

Standard PCA finds components that maximize total variance. For neural population data recorded across multiple experimental conditions, this is often unsatisfying: PCA components typically mix time-modulated activity with stimulus-modulated activity, making interpretation difficult.

**Goal of dPCA**: find a small number of components that each capture variance attributable to a *single* task parameter (stimulus, time, their interaction, etc.)—while collectively explaining as much variance as possible.

------------------------------------------------------------------------

## 2. Data Structure and Notation

| Symbol | Meaning |
|----------------------------------|--------------------------------------|
| $N$ | number of neurons (first axis) |
| $K$ | number of task-parameter axes |
| $d_k$ | size of the $k$-th parameter axis |
| $X \in \mathbb{R}^{N \times d_1 \times \cdots \times d_K}$ | trial-averaged population data |
| $\phi \subseteq \{1,\ldots,K\}$ | a subset of parameter axes (a "marginalization") |
| $X_\phi$ | marginalization of $X$ corresponding to $\phi$ |

**Centered data**: subtract per-neuron mean across all conditions: $$\tilde{X}_{n,i_1,\ldots,i_K} = X_{n,i_1,\ldots,i_K} - \frac{1}{d_1 \cdots d_K} \sum_{j_1,\ldots,j_K} X_{n,j_1,\ldots,j_K}$$

------------------------------------------------------------------------

## 3. Marginalizations: ANOVA Decomposition

The key idea is an **ANOVA-style additive decomposition** of the centered data $\tilde{X}$.

### 3.1 Two-parameter example ($K = 2$, labels = "st")

With stimulus axis $s$ (size $S$) and time axis $t$ (size $T$):

$$\tilde{X}_{n,s,t} = \underbrace{X^{(s)}_{n,s,t}}_{\text{stimulus}} + \underbrace{X^{(t)}_{n,s,t}}_{\text{time}} + \underbrace{X^{(st)}_{n,s,t}}_{\text{interaction}}$$

where:

$$X^{(s)}_{n,s,t} = \frac{1}{T}\sum_t \tilde{X}_{n,s,t}
\quad\text{(mean over time, constant in }t\text{)}$$

$$X^{(t)}_{n,s,t} = \frac{1}{S}\sum_s \tilde{X}_{n,s,t}
\quad\text{(mean over stimuli, constant in }s\text{)}$$

$$X^{(st)}_{n,s,t} = \tilde{X}_{n,s,t} - X^{(s)}_{n,s,t} - X^{(t)}_{n,s,t}
\quad\text{(residual interaction)}$$

**Verification**: $X^{(s)} + X^{(t)} + X^{(st)} = \tilde{X}$ ✓

### 3.2 ANOVA analogy

This is the 2-way ANOVA decomposition from statistics, applied to each neuron $n$ independently:

| Term | ANOVA name | Interpretation |
|-----------------|-----------------------|---------------------------------|
| $X^{(s)}$ | "main effect of stimulus" | activity that varies with stimulus but not time |
| $X^{(t)}$ | "main effect of time" | activity that varies with time but not stimulus |
| $X^{(st)}$ | "interaction" | activity that depends on both simultaneously |

### 3.3 Orthogonality

The marginalizations are **Frobenius-orthogonal**: $$\langle X^{(\phi)}, X^{(\psi)} \rangle_F = \sum_{n,i_1,\ldots,i_K} X^{(\phi)}_{n,i_1,\ldots,i_K} \cdot X^{(\psi)}_{n,i_1,\ldots,i_K} = 0
\quad \text{for } \phi \neq \psi$$

This follows directly from the inclusion-exclusion construction.

**Consequence**: the marginal variances partition the total variance: $$\sum_\phi \|X^{(\phi)}\|_F^2 = \|\tilde{X}\|_F^2$$

### 3.4 General K-parameter case

For $K$ parameters, each subset $\phi \subseteq \{1,\ldots,K\}$ defines a marginalization built by inclusion-exclusion:

$$X^{(\phi)} = \text{mean}_{\phi^c}(\tilde{X}) - \sum_{\psi \subsetneq \phi} X^{(\psi)}$$

where $\phi^c = \{1,\ldots,K\} \setminus \phi$ is the complement and "$\text{mean}_{\phi^c}$" denotes averaging over all axes in $\phi^c$ (holding axes in $\phi$ fixed, broadcasting the result).

There are $2^K - 1$ non-empty marginalizations (the empty set is absorbed into centering).

------------------------------------------------------------------------

## 4. The Optimization Problem

### 4.1 Per-marginalization objective

For each marginalization $\phi$, dPCA seeks an **encoder** $F_\phi$ and **decoder** $D_\phi$, both $N \times k$ matrices ($k \ll N$), that minimize the reconstruction error of $X^{(\phi)}$ from the full data $X$:

$$\mathcal{L}_\phi(F_\phi, D_\phi) = \left\| X^{(\phi)} - F_\phi D_\phi^\top \tilde{X} \right\|_F^2$$

Stacking over all marginalizations:

$$\min_{F_\phi, D_\phi} \sum_\phi \left\| X^{(\phi)} - F_\phi D_\phi^\top \tilde{X} \right\|_F^2$$

### 4.2 Why minimize reconstruction of $X^{(\phi)}$ from $X$?

The term $D_\phi^\top \tilde{X}$ is the latent projection (low-dimensional representation). The term $F_\phi D_\phi^\top \tilde{X}$ is the reconstruction back to neural space.

By minimizing $\|X^{(\phi)} - F_\phi D_\phi^\top \tilde{X}\|_F^2$, we ask: \> "Find $k$ dimensions of the full data $\tilde{X}$ that best reconstruct the \> $\phi$-specific component of the data."

This is *demixed* because $D_\phi^\top \tilde{X}$ is forced to encode primarily the $\phi$ component, not other components.

------------------------------------------------------------------------

## 5. Closed-Form Solution

### 5.1 Optimal decoder $D_\phi$

Minimizing $\mathcal{L}_\phi$ over $F_\phi$ given $D_\phi$:

$$F_\phi = X^{(\phi)} \tilde{X}^\dagger D_\phi \cdot (D_\phi^\top \tilde{X} \tilde{X}^\top D_\phi)^{-1} \tilde{X} \tilde{X}^\top D_\phi \cdot \ldots$$

This simplifies (see Kobak 2016, Appendix) to:

Define the "bridge matrix": $$C_\phi = X_\phi^{\text{flat}} \cdot (\tilde{X}^{\text{flat}})^\dagger \in \mathbb{R}^{N \times N}$$

where superscript "flat" means reshaping to $[N, d_1 \cdots d_K]$ and $^\dagger$ is the Moore-Penrose pseudoinverse.

Then the optimal $F_\phi$ (encoder) and $D_\phi$ (decoder) are given by:

$$\text{SVD of } C_\phi \tilde{X}^{\text{flat}} = U \Sigma V^\top, \quad \text{(truncated to rank }k\text{)}$$

$$\boxed{F_\phi = U} \qquad \boxed{D_\phi = C_\phi^\top U}$$

More explicitly: $D_\phi = (C_\phi^\top U)$.

### 5.2 Intuition for $C_\phi$

$C_\phi = X^{(\phi)} \tilde{X}^\dagger$ answers: "How does the marginalization $X^{(\phi)}$ arise as a linear map from the full data $\tilde{X}$?"

If $X^{(\phi)} = \tilde{X}$, then $C_\phi = I$ (identity) and dPCA reduces to standard PCA.

If $X^{(\phi)}$ is purely due to stimulus and the data contains both stimulus and time components, then $C_\phi$ projects out the non-stimulus variance.

### 5.3 Why $F_\phi \neq D_\phi$ (non-symmetric)

In standard PCA, encoder = decoder = principal directions.\
In dPCA, they differ:

- $D_\phi$ (decoder) = the direction in $\tilde{X}$ to project onto\
  $\Rightarrow$ minimizes reconstruction error of $X^{(\phi)}$
- $F_\phi$ (encoder) = the direction in neural space to reconstruct to\
  $\Rightarrow$ the "readout" that best reconstructs $X^{(\phi)}$ from $Z_\phi$

The asymmetry $F_\phi \neq D_\phi$ means that the latent code $Z_\phi = D_\phi^\top \tilde{X}$ is not simply a projection of $X^{(\phi)}$ back onto itself—it is a projection of the full data $\tilde{X}$ that captures the $\phi$-specific structure.

### 5.4 The transform and inverse transform

**Transform** (encoding): $$Z_\phi = D_\phi^\top \tilde{X}^{\text{flat}} \in \mathbb{R}^{k \times (d_1 \cdots d_K)}$$

**Inverse transform** (decoding): $$\hat{X}^{(\phi)} = F_\phi Z_\phi = F_\phi D_\phi^\top \tilde{X}^{\text{flat}} \in \mathbb{R}^{N \times (d_1 \cdots d_K)}$$

------------------------------------------------------------------------

## 6. Explained Variance

For marginalization $\phi$ and component $j \in \{1,\ldots,k\}$:

$$R^2_{\phi,j} = \frac{\| D_{\phi,j}^\top \tilde{X}^{\text{flat}} \|_2^2}{\| \tilde{X}^{\text{flat}} \|_F^2}$$

where $D_{\phi,j}$ is the $j$-th column of $D_\phi$.

This measures how much of the **total** variance is captured by the $j$-th component of the $\phi$-marginalization.

Note: explained variances across all marginalizations and components can exceed 1.0 (they are not a partition), because different decoder columns $D_{\phi,j}$ are not globally orthogonal.

------------------------------------------------------------------------

## 7. Regularization

For neural data with many neurons ($N$ large) and limited conditions ($d_1
\cdots d_K$ small), the pseudoinverse $(\tilde{X}^{\text{flat}})^\dagger$ can be ill-conditioned. Tikhonov regularization adds a ridge term:

$$\tilde{X}^{\text{reg}} = \left[\tilde{X}^{\text{flat}} \;\Big|\; \lambda I_N \right] \in \mathbb{R}^{N \times (d + N)}$$

and similarly appends zero columns to each $X^{(\phi)}^{\text{flat}}$.

This penalizes large norms of the decoder:

$$\min_{F, D} \| X^{(\phi)}_{\text{flat}} - F D^\top \tilde{X}_{\text{flat}} \|_F^2 + \lambda^2 \|F\|_F^2$$

The optimal $\lambda$ can be found by cross-validation (set `regularizer="auto"` in Python or tune `regularizer` in R).

------------------------------------------------------------------------

## 8. Significance Testing

### 8.1 Procedure

1.  **True score**: for each component $j$ of marginalization $\phi$:
    - Split trials into train/test (repeat $n_\text{splits}$ times)\
    - Fit dPCA on train, project test\
    - Compute classification accuracy over the $\phi$-axis (nearest-mean classifier)\
    - Average over time points (or evaluate per time point for temporal significance)
2.  **Null distribution**: shuffle condition labels $n_\text{shuffles}$ times:
    - For each shuffle, repeat the above classification\
    - Record the maximum score across shuffles
3.  **Significance**: a component is significant at a time point if\
    $\text{true\_score}[j,t] > \max_\text{shuffle}[\text{shuffled\_score}[j,t]]$

This is equivalent to a permutation test at level $p < 1/n_\text{shuffles}$.

### 8.2 n_consecutive filter

Single time-points can be significant by chance. Requiring $n_\text{consecutive}$ consecutive significant time-points reduces false positives.

------------------------------------------------------------------------

## 9. Comparison with Related Methods

### 9.1 Standard PCA

|   | PCA | dPCA |
|----------------|-------------------------|-------------------------------|
| Objective | maximize total variance | maximize marginalization-specific variance |
| Components | maximize $\|U^\top X\|_F^2$ | maximize $\|D_\phi^\top \tilde{X}\|_F^2$ for $X^{(\phi)}$ |
| Encoder = Decoder? | yes | no (in general) |
| Number of decomps | 1 | $2^K - 1$ |

The key difference: PCA seeks directions that capture variance from *any* source; dPCA seeks directions specific to one source each.

### 9.2 Linear Discriminant Analysis (LDA)

LDA also separates stimulus effects from noise. However: - LDA requires discrete class labels; dPCA handles continuous parameters - LDA maximizes between-class / within-class variance ratio (Fisher criterion) - dPCA minimizes reconstruction error of marginalizations

### 9.3 Factor Analysis

Factor analysis models $X = L F + \epsilon$ with noise term $\epsilon$. dPCA has no explicit noise model; it is a purely geometric decomposition.

### 9.4 Connection to jPCA

jPCA (Churchland et al. 2012) looks for rotational dynamics in the time-marginalization $X^{(t)}$:

- **First apply dPCA**: project onto $Z^{(t)} = D_t^\top \tilde{X}$\
  (captures time-varying component, demixed from stimulus)
- **Then apply jPCA** to $Z^{(t)}$: find rotational dynamics within the demixed time subspace

This is the "dpca_jpca_pipeline" defined in ARCHITECTURE.md Phase 3.

------------------------------------------------------------------------

## 10. Implementation Notes (`demixedPCA_lib.R`)

### 10.1 Array conventions

- R arrays are **column-major** (first index varies fastest)
- Python numpy arrays are **row-major** (last index varies fastest)
- This affects flattening: `matrix(X, nrow=N)` in R ≠ `X.reshape(N,-1)` in Python
- The math is equivalent; the element ordering within the flat matrix differs, but all operations are consistent

### 10.2 broadcast_to in R

R does not support numpy-style broadcasting for multi-dimensional arrays. `expand_to(M, target_dim)` in `demixedPCA_lib.R` implements this by: 1. Identifying singleton dimensions 2. Moving singleton to last position (`aperm`) 3. Replicating with `rep()` 4. Moving back with `aperm(., order(perm))`

### 10.3 `apply` and dimension ordering

`apply(A, MARGIN, mean)` returns an array with dimensions **in the order they appear in MARGIN**. When MARGIN = `setdiff(1:(K+1), ax_r)` (sorted), the result has dims in ascending order = same as original (except the averaged dimension is missing). `array(as.vector(M), dim = new_dim)` then correctly inserts a singleton at `ax_r`.

### 10.4 Computing $(\tilde{X}^{\text{flat}})^\dagger$

`MASS::ginv(X_flat)` computes the Moore-Penrose pseudoinverse. For $X \in \mathbb{R}^{N \times P}$ with $P > N$:

$$X^\dagger = X^\top (X X^\top)^{-1}$$

This is computed via SVD internally. With regularization: $(XX^\top + \lambda^2 I)^{-1} X$ is more stable.

------------------------------------------------------------------------

## 11. Toy Example Walkthrough

Data: $N=3$ neurons, $S=2$ stimuli, $T=3$ time points.

```         
X[1,s,t]:
  s=1: [1, 2, 3]   (ramps with time; neuron 1 modulated by time)
  s=2: [2, 3, 4]   (shifts up with stimulus)

X[2,s,t]:
  s=1: [1, 1, 1]   (flat; neuron 2 modulated by stimulus only)
  s=2: [3, 3, 3]

X[3,s,t]:
  s=1: [0, 1, -1]  (no systematic modulation)
  s=2: [0, -1, 1]
```

After centering (subtract mean over s,t per neuron):\
- Neuron 1 mean = (1+2+3+2+3+4)/6 = 2.5 - Neuron 2 mean = (1+1+1+3+3+3)/6 = 2 - Neuron 3 mean = 0

Marginalization $X^{(s)}$ = mean over T: - Neuron 1: s=1 → mean(\[1,2,3\])-2.5 = -0.5, s=2 → +0.5 - Neuron 2: s=1 → -1, s=2 → +1 - Neuron 3: s=1 → 0, s=2 → 0 (no stimulus effect)

Marginalization $X^{(t)}$ = mean over S: - Neuron 1: t=1 → 1.5-2.5=-1, t=2 → 2.5-2.5=0, t=3 → 3.5-2.5=+1 - Neuron 2: t=1,2,3 → 2-2=0 (no time effect) - Neuron 3: t=1 → 0, t=2 → 0, t=3 → 0

dPCA on $X^{(s)}$: finds that neurons 1 and 2 co-vary with stimulus → single component captures both. The decoder $D_s$ points in the stimulus direction.

dPCA on $X^{(t)}$: finds neuron 1's ramp = single time component. The decoder $D_t$ points primarily at neuron 1.

------------------------------------------------------------------------

## 12. Summary of Key Mathematical Relationships

$$\boxed{X^{(\phi)} = \text{mean}_{\phi^c}(\tilde{X}) - \sum_{\psi \subsetneq \phi} X^{(\psi)}}$$

$$\boxed{C_\phi = X^{(\phi)}_\text{flat} \cdot \tilde{X}_\text{flat}^\dagger}$$

$$\boxed{C_\phi \tilde{X}_\text{flat} = U_\phi \Sigma_\phi V_\phi^\top \quad \Rightarrow \quad F_\phi = U_\phi,\quad D_\phi = C_\phi^\top U_\phi}$$

$$\boxed{Z_\phi = D_\phi^\top \tilde{X}_\text{flat}}$$

$$\boxed{R^2_{\phi,j} = \frac{\|D_{\phi,j}^\top \tilde{X}_\text{flat}\|^2}{\|\tilde{X}_\text{flat}\|_F^2}}$$