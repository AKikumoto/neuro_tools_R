# jPCA — Mathematical Notes

**Reference**: Churchland, M.M. et al. (2012). Neural population dynamics during reaching. *Nature* 487, 51–56.  
**R implementation**: `jPCA_lib.R`  
**Visual supplement**: [jPCA_geometry.html](jPCA_geometry.html)

---

## 1. The Question

EEG decoding asks: "Can we decode *what* the brain represents at time $t$?"  
jPCA asks a different question:

> Is the *change* in population state itself structured — specifically, rotational?

The **population state** at time $t$ is an $N$-dimensional vector $\mathbf{x}(t) \in \mathbb{R}^N$
stacking activity across $N$ channels, tracing a trajectory through $N$-dimensional space over time.

---

## 2. Linear Dynamics Assumption

Simplest assumption: state change is proportional to current state:

$$\dot{\mathbf{x}}(t) = M\,\mathbf{x}(t)$$

$\dot{\mathbf{x}}$ is approximated by finite differences $\mathbf{x}(t+1) - \mathbf{x}(t)$.  
Unconstrained $M$ allows rotations, expansions, and shears — too general to test for rotation specifically.  
jPCA constrains $M$ to be **skew-symmetric**.

---

## 3. Skew-Symmetric Matrices and Rotation

### Definition

$$M_\text{skew} = -M_\text{skew}^\top \quad (m_{ij} = -m_{ji},\; m_{ii} = 0)$$

### Why skew-symmetric means rotation

The solution to $\dot{\mathbf{x}} = M_\text{skew}\,\mathbf{x}$ is $\mathbf{x}(t) = e^{M_\text{skew}\,t}\,\mathbf{x}(0)$.

Since $M_\text{skew}$ is skew-symmetric:

$$\bigl(e^{M_\text{skew}}\bigr)^\top = e^{-M_\text{skew}} = \bigl(e^{M_\text{skew}}\bigr)^{-1}$$

→ $e^{M_\text{skew}}$ is orthogonal (rotation matrix): $\|\mathbf{x}(t)\| = \|\mathbf{x}(0)\|$ for all $t$.

### Eigenvalues are purely imaginary

For $M_\text{skew}\,\mathbf{v} = \lambda\mathbf{v}$, multiply left by $\bar{\mathbf{v}}^\top$:

$$\bar{\mathbf{v}}^\top M_\text{skew}\,\mathbf{v} = \lambda|\mathbf{v}|^2$$

Taking the complex conjugate of the left side and using $M_\text{skew}^\top = -M_\text{skew}$ shows this quantity is purely imaginary → $\lambda = \pm i\omega$ ($\omega \in \mathbb{R}$).

The corresponding solution $e^{i\omega t} = \cos(\omega t) + i\sin(\omega t)$ is rotation at angular frequency $\omega$.

---

## 4. Rotation Strength via Angle θ

After projecting onto the jPC plane, compute for each (condition $c$, time $t$):

$$\theta(t,c) = \operatorname{atan2}\!\bigl(\mathbf{x} \wedge \dot{\mathbf{x}},\; \mathbf{x} \cdot \dot{\mathbf{x}}\bigr)$$

where $\mathbf{x} \cdot \dot{\mathbf{x}}$ is the dot product and $\mathbf{x} \wedge \dot{\mathbf{x}} = x_1\dot{x}_2 - x_2\dot{x}_1$ is the 2D cross product.

| $\theta$ | Geometry |
|---|---|
| $\approx +\pi/2$ | Pure counter-clockwise rotation ($\dot{\mathbf{x}} \perp \mathbf{x}$) |
| $\approx 0$ | Pure expansion ($\dot{\mathbf{x}} \parallel \mathbf{x}$) |
| $\approx -\pi/2$ | Pure clockwise rotation |
| $\approx \pm\pi$ | Pure contraction |

A histogram of $\theta$ peaked at $\pi/2$ indicates rotational dynamics.

**Implementation note**: with `omega=0.2` surrogate data the empirical peak is ≈ 1.775 rather than $\pi/2 \approx 1.571$. This is a systematic bias from (1) finite-difference phase lag (+0.1 rad) and (2) elliptical distortion from unequal PC variances (+0.1 rad). It is not noise — see [resource/jPCA_test_results_en.md](jPCA_test_results_en.md) Test 7.

---

## 5. Rotation Strength via R² Ratio

$$R^2_\text{ratio} = \frac{R^2_\text{skew}}{R^2_\text{unrestr}}, \qquad R^2 = 1 - \frac{\|dX - MX_\text{prev}\|_F^2}{\|dX\|_F^2}$$

| Value | Interpretation |
|---|---|
| $\approx 1.0$ | Dynamics are nearly pure rotation |
| $\approx 0.5$ | Expected for random data |
| $\ll 0.5$ | Should not occur ($M_\text{skew}$ is a subset of unconstrained solutions) |

Random baseline at 0.5: any matrix $\hat{M}$ decomposes orthogonally into symmetric $S$ and skew-symmetric $A$ parts, which on average carry equal explanatory power.

**Note**: `normalize=FALSE` can yield negative $R^2_\text{ratio}$ when PC variances differ greatly. `normalize=TRUE` (default) is required for stable estimates.

---

## 6. Algorithm

```
Input: X_list — list of [N × T] matrices, one per condition (≥ 3 conditions)

Step 1: PCA pre-processing
  (a) Concatenate conditions: X_full [N × C*T]
  (b) Subtract cross-condition mean at each timepoint
      (removes shared evoked response that would be mis-detected as rotation)
  (c) PCA → retain top n_pcs PCs  (default 6)
  (d) Normalise each PC by its standard deviation
      (equalises scales across PCs for stable M_skew estimation)

Step 2: Finite differences
  dX     = X[t+1] − X[t]    [n_pcs × C*(T-1)]
  X_prev = X[1:(T-1)]

Step 3: Unconstrained least-squares
  M_hat = dX X_prev^T (X_prev X_prev^T)^{-1}

Step 4: Project to skew-symmetric
  M_skew = (M_hat − M_hat^T) / 2
  (This is the closest skew-symmetric matrix to M_hat in Frobenius norm)

Step 5: Eigendecomposition
  eigen(M_skew) → λ = ±iω, V   (purely imaginary eigenvalues)
  Sort by |ω| descending

Step 6: Recover real rotation plane
  For leading complex conjugate pair (v₁, v̄₁):
    jPC1 = Re(v₁) / ||Re(v₁)||
    jPC2 = −Im(v₁) / ||Im(v₁)||
  These are orthogonal (Re and Im of skew-symmetric eigenvectors are perpendicular)

Step 7: Project data
  W = [jPC1; jPC2]  [2 × n_pcs]
  X_jPCA = W X_red  [2 × C*T]
```

### Why jPC1 ⊥ jPC2

Writing $\mathbf{v}_1 = \mathbf{a} + i\mathbf{b}$, the inner product $\mathbf{a}^\top\mathbf{b}$ equals zero because the real and imaginary parts of a complex eigenvector of a real skew-symmetric matrix are orthogonal.

---

## 7. References

- Churchland, M.M. et al. (2012). Neural population dynamics during reaching. *Nature* 487, 51–56.  
  Fig. 3: rotation trajectories. Fig. 6: θ histograms. Supplementary Methods: algorithm details.
