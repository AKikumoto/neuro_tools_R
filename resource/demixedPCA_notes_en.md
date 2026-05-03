# demixed PCA (dPCA) — Mathematical Notes

**Reference**: Kobak, D. et al. (2016). Demixed principal component analysis of neural population data. *eLife* 5:e10989.  
**R implementation**: `demixedPCA_lib.R`  
**Visual supplement**: [dPCA_anova_decomp.html](dPCA_anova_decomp.html)

---

## 1. Motivation

Standard PCA maximises total variance, so its components mix contributions from all task parameters.  
**dPCA** separates them: it finds components that each capture variance attributable to a *single* task parameter (stimulus, time, rule, or their interactions).

---

## 2. Data Structure

| Symbol | Meaning |
|--------|---------|
| $N$ | number of channels |
| $K$ | number of task-parameter axes |
| $d_k$ | size of the $k$-th axis |
| $X \in \mathbb{R}^{N \times d_1 \times \cdots \times d_K}$ | trial-averaged data |
| $\phi \subseteq \{1,\ldots,K\}$ | a subset of axes ("marginalization") |

**Centering**: subtract the per-channel mean across all conditions.

---

## 3. ANOVA Decomposition (Marginalization)

The centred data $\tilde{X}$ decomposes additively — exactly as in a multi-way ANOVA:

$$\tilde{X} = \sum_{\phi \neq \emptyset} X^{(\phi)}$$

### Two-parameter example ($K=2$, labels = "st")

$$X^{(s)}_{n,s,t} = \frac{1}{T}\sum_t \tilde{X}_{n,s,t} \qquad \text{(mean over time; varies with $s$ only)}$$

$$X^{(t)}_{n,s,t} = \frac{1}{S}\sum_s \tilde{X}_{n,s,t} \qquad \text{(mean over stimulus; varies with $t$ only)}$$

$$X^{(st)}_{n,s,t} = \tilde{X}_{n,s,t} - X^{(s)}_{n,s,t} - X^{(t)}_{n,s,t} \qquad \text{(residual interaction)}$$

### General $K$-parameter formula (inclusion–exclusion)

$$X^{(\phi)} = \operatorname{mean}_{\phi^c}(\tilde{X}) - \sum_{\psi \subsetneq \phi} X^{(\psi)}$$

where $\phi^c$ denotes the complement of $\phi$ and "mean$_{\phi^c}$" averages over all axes in $\phi^c$. There are $2^K - 1$ non-empty marginalizations; compute them from smallest to largest subsets.

### Frobenius Orthogonality

$$\langle X^{(\phi)}, X^{(\psi)} \rangle_F = 0 \quad \text{for } \phi \neq \psi$$

This follows from the inclusion–exclusion construction (same principle as ANOVA non-confounding).  
**Consequence**: marginal variances partition total variance: $\sum_\phi \|X^{(\phi)}\|_F^2 = \|\tilde{X}\|_F^2$.

---

## 4. Optimization Problem

For each marginalization $\phi$, find encoder $F_\phi$ and decoder $D_\phi$ (both $N \times k$) minimising:

$$\mathcal{L}_\phi = \left\| X^{(\phi)} - F_\phi D_\phi^\top \tilde{X} \right\|_F^2$$

The combined objective is $\sum_\phi \mathcal{L}_\phi$.  
This asks: "find $k$ directions in $\tilde{X}$ whose projection best reconstructs the $\phi$-specific component."

---

## 5. Closed-Form Solution

### Bridge matrix

$$C_\phi = X^{(\phi)}_\text{flat} \cdot \tilde{X}_\text{flat}^\dagger \in \mathbb{R}^{N \times N}$$

where "flat" reshapes to $[N, d_1 \cdots d_K]$ and $^\dagger$ is the Moore–Penrose pseudoinverse.

**Intuition**: $C_\phi$ encodes how $X^{(\phi)}$ arises as a linear map of $\tilde{X}$.  
If $X^{(\phi)} = \tilde{X}$, then $C_\phi = I$ and dPCA reduces to standard PCA.

### Optimal encoder and decoder

$$\text{SVD of }C_\phi\,\tilde{X}_\text{flat} = U_\phi\Sigma_\phi V_\phi^\top \quad \Rightarrow \quad \boxed{F_\phi = U_\phi,\quad D_\phi = C_\phi^\top U_\phi}$$

### Asymmetry $F_\phi \neq D_\phi$

In standard PCA encoder = decoder = principal directions.  
In dPCA:
- $D_\phi$ (decoder): direction in $\tilde{X}$ to project onto → maximises $\phi$-specific variance
- $F_\phi$ (encoder): direction in neural space to reconstruct to → best reconstructs $X^{(\phi)}$ from $Z_\phi$

This asymmetry is the core of "demixing": the latent code $Z_\phi = D_\phi^\top\tilde{X}$ is forced to encode primarily $\phi$, not other components.

**Biorthogonality**: $D_\phi^\top F_\phi \approx I_k$ — each latent dimension corresponds to exactly one factor.

---

## 6. Transform and Inverse Transform

$$Z_\phi = D_\phi^\top \tilde{X}_\text{flat} \in \mathbb{R}^{k \times (d_1 \cdots d_K)} \quad \text{reshape to } [k \times d_1 \times \cdots \times d_K]$$

$$\hat{X}^{(\phi)} = F_\phi Z_\phi \in \mathbb{R}^{N \times (d_1 \cdots d_K)}$$

---

## 7. Explained Variance

$$R^2_{\phi,j} = \frac{\|D_{\phi,j}^\top \tilde{X}_\text{flat}\|^2}{\|\tilde{X}_\text{flat}\|_F^2}$$

Summing over all $(\phi, j)$ can exceed 1.0 because $D_{\phi,j}$ are not globally orthogonal across marginalizations.

---

## 8. Regularization

When $N > d_1 \cdots d_K$ (more channels than conditions × time), the pseudoinverse is ill-conditioned.  
Append $\lambda I_N$ columns: $\tilde{X}^\text{reg} = [\tilde{X}_\text{flat} \;|\; \lambda I_N]$, equivalently minimising $\mathcal{L}_\phi + \lambda^2\|F_\phi\|_F^2$.

For typical EEG ($N = 150$, $T \times C = 250 \times 6 = 1500$), $N < d_1 \cdots d_K$ and regularization is not needed.

---

## 9. Connection to jPCA

Apply jPCA within the time marginalization:

```r
Z_t <- dpca_transform(X, dpca_model)[["t"]]   # [k × S × T]
# reshape to per-condition list, then:
jpca_model <- jpca_fit(Z_t_list)
```

This asks: "Does the *time-specific* representation rotate?" — with stimulus variance already removed by dPCA.

---

## 10. Key Formulas

$$X^{(\phi)} = \operatorname{mean}_{\phi^c}(\tilde{X}) - \sum_{\psi \subsetneq \phi} X^{(\psi)}$$

$$C_\phi = X^{(\phi)}_\text{flat} \cdot \tilde{X}_\text{flat}^\dagger$$

$$C_\phi\,\tilde{X}_\text{flat} = U\Sigma V^\top \;\Rightarrow\; F_\phi = U,\; D_\phi = C_\phi^\top U$$

$$Z_\phi = D_\phi^\top\tilde{X}_\text{flat}$$

$$R^2_{\phi,j} = \|D_{\phi,j}^\top\tilde{X}_\text{flat}\|^2 \;/\; \|\tilde{X}_\text{flat}\|_F^2$$

---

## 11. References

- Kobak, D. et al. (2016). Demixed principal component analysis of neural population data. *eLife* 5:e10989.
- Brendel, W. et al. (2011). Demixed principal component analysis. *NIPS 2011*.
- machenslab/dPCA (Python/MATLAB reference implementation): https://github.com/machenslab/dPCA
