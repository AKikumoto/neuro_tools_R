# Reduced Rank Regression — Mathematical Notes

**Reference**: Wu B & Pillow JW (2025). *Reduced rank regression for neural communication: a tutorial for neuroscientists.* arXiv:2512.12467v1.

**Python implementation**: `original/RRR/python/fitting.py`\
**R implementation**: `R/RRR_lib.R` (this project)

------------------------------------------------------------------------

## 1. Motivation

Standard multivariate regression is full-rank: when predicting $n$ output neurons from $m$ input neurons, it fits an unconstrained $m \times n$ weight matrix with $mn$ free parameters. This overfits when $T$ (the number of time samples) is not large relative to $m$ or $n$, and it offers no interpretable structure.

**The communication subspace hypothesis** (Semedo et al. 2019): only $r \ll \min(m,n)$ dimensions of input-region activity drive output-region activity. The input sends a low-dimensional signal; the output reads it out in a low-dimensional subspace. This is the **communication rank** $r$.

**Why RRR and not PCA?** PCA of the input region finds the dimensions of maximum variance within that region. Those may not be the dimensions that communicate to the output. RRR is *supervised*: it uses the cross-region relationship to define the subspace, not within-region variance alone.

**Goal**: find $W = UV^\top \in \mathbb{R}^{m \times n}$ of rank $r$ that minimizes $\|Y - XW\|_F^2$, recovering both the input axes $U$ (which directions of $x_t$ are transmitted) and the output axes $V$ (which directions of $y_t$ are driven by input).

------------------------------------------------------------------------

## 2. Data Structure and Notation

| Symbol | Meaning |
|------------------------------------------------|-----------------------------------------------|
| $T$ | number of time samples (rows of data) |
| $m$ | number of input neurons (columns of $X$) |
| $n$ | number of output neurons (columns of $Y$) |
| $r$ | communication rank, $r \ll \min(m,n)$ |
| $X \in \mathbb{R}^{T \times m}$ | input activity matrix (centered) |
| $Y \in \mathbb{R}^{T \times n}$ | output activity matrix (centered) |
| $W \in \mathbb{R}^{m \times n}$ | full weight matrix |
| $U \in \mathbb{R}^{m \times r}$ | input axes (columns = input communication directions) |
| $V \in \mathbb{R}^{n \times r}$ | output axes, $V^\top V = I_r$ (semi-orthogonal) |
| $W_\text{LS} \in \mathbb{R}^{m \times n}$ | ordinary least-squares (full-rank) estimate |
| $W_\text{ridge}$ | ridge-regularized estimate |
| $\Sigma_X = \frac{1}{T} X^\top X$ | input covariance $[m \times m]$ |
| $\Sigma_Y = \frac{1}{T} Y^\top Y$ | output covariance $[n \times n]$ |
| $\Sigma \in \mathbb{R}^{n \times n}$ | output noise covariance (full-covariance model) |
| $\lambda$ | ridge penalty coefficient |
| $\|\cdot\|_F$ | Frobenius norm: $\|A\|_F^2 = \text{Tr}[A^\top A]$ |

**Centered data**: subtract per-column means before fitting. Without centering, the mean acts as an extra communication dimension (inflates effective rank by 1).

------------------------------------------------------------------------

## 3. The Linear-Gaussian Model

The generative model for a single time step is [eq. 1--2]:

$$y_t = W^\top x_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma^2 I_n)$$

Stacking $T$ time steps into matrices:

$$\boxed{Y \approx XW + E, \qquad W \in \mathbb{R}^{m \times n}}$$

where $E$ collects the i.i.d. noise terms and each row of $Y$ is a transposed observation $y_t^\top$.

**Low-rank constraint**: restrict $W$ to rank $r$:

$$W = UV^\top, \quad U \in \mathbb{R}^{m \times r},\; V \in \mathbb{R}^{n \times r},\; V^\top V = I_r$$

**Geometric interpretation**:

- Columns of $U$: input axes. Each column $u_k$ is a direction in input space $\mathbb{R}^m$ that contributes to the output. The scalar $u_k^\top x_t$ is the $k$-th latent communication signal at time $t$.
- Columns of $V$: output axes. Each column $v_k$ is the direction in output space $\mathbb{R}^n$ driven by that signal.
- The *private dimensions* of the input are $\{x : U^\top x = 0\}$, the orthogonal complement of the column space of $U$. Activity in private dimensions is not communicated.

**Why semi-orthogonal $V$?** The constraint $V^\top V = I$ removes the rotational ambiguity: for any invertible $r \times r$ matrix $A$, $(UA^{-\top})(VA)^\top = UV^\top$, so infinitely many $(U,V)$ pairs represent the same $W$. Fixing $V^\top V = I$ makes the factorization unique (up to column sign flips).

------------------------------------------------------------------------

## 4. Least Squares (Full-Rank) Estimate

To derive the unconstrained minimizer of $\|Y - XW\|_F^2$, differentiate with respect to $W$ and set to zero:

$$\frac{\partial}{\partial W} \|Y - XW\|_F^2 = -2 X^\top(Y - XW) = 0$$

$$\Rightarrow X^\top X W = X^\top Y$$

$$\boxed{W_\text{LS} = (X^\top X)^{-1} X^\top Y} \tag{eq. 6}$$

This exists when $X^\top X$ is invertible, which requires $T \geq m$ and full column rank of $X$.

**Geometric note**: $XW_\text{LS} = X(X^\top X)^{-1} X^\top Y = P_X Y$, the orthogonal projection of $Y$ onto the column space of $X$. The OLS fit is therefore the best linear predictor of $Y$ from $X$ with no rank constraint.

**Overfitting**: when $T$ is not large relative to $m$, $X^\top X$ is ill-conditioned and $W_\text{LS}$ fits noise. The OLS $R^2$ on held-out data will be lower than on training data. This is the primary motivation for both rank restriction (RRR) and ridge regularization.

------------------------------------------------------------------------

## 5. SVD Review

Any matrix $A \in \mathbb{R}^{p \times q}$ (with $p \geq q$) has the **singular value decomposition**:

$$A = U_A S_A V_A^\top, \quad U_A \in \mathbb{R}^{p \times q},\; S_A = \text{diag}(\sigma_1,\ldots,\sigma_q),\; V_A \in \mathbb{R}^{q \times q}$$

with $U_A^\top U_A = I$, $V_A^\top V_A = V_A V_A^\top = I$, and $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$.

**Eckart-Young theorem** (best rank-$r$ approximation) [eq. 8]: the rank-$r$ matrix closest to $A$ in Frobenius norm is:

$$\hat{A}_r = \sum_{k=1}^{r} \sigma_k u_k v_k^\top$$

where $u_k$, $v_k$ are the $k$-th left and right singular vectors. Formally:

$$\hat{A}_r = \arg\min_{\text{rank}(B) \leq r} \|A - B\|_F^2$$

**Preview — why SVD of $W_\text{LS}$ is not the RRR solution**: the Eckart-Young theorem minimizes $\|W_\text{LS} - W\|_F^2$ (error in the weight matrix). RRR minimizes $\|Y - XW\|_F^2$ (error in the predictions). These are different objectives unless $X^\top X \propto I$. Section 7 develops this distinction in full.

------------------------------------------------------------------------

## 6. The RRR Estimator

### 6.1 The Three-Step Algorithm

Given $X$, $Y$, and rank $r$, the RRR estimator is [eqs. 14--17]:

**Step 1**: Compute the full-rank OLS estimate:

$$W_\text{LS} = (X^\top X)^{-1} X^\top Y \tag{eq. 14}$$

**Step 2**: Compute the top $r$ eigenvectors of $W_\text{LS}^\top X^\top X W_\text{LS}$:

$$W_\text{LS}^\top X^\top X W_\text{LS} V_r = V_r \Lambda_r \tag{eq. 15}$$

where $V_r \in \mathbb{R}^{n \times r}$, $V_r^\top V_r = I_r$, and $\Lambda_r$ contains the top $r$ eigenvalues.

**Step 3**: Project $W_\text{LS}$ onto the subspace spanned by $V_r$:

$$\boxed{W_\text{RRR} = W_\text{LS} V_r V_r^\top} \tag{eq. 16}$$

### 6.2 Factored Form

The factored form follows immediately from Step 3 [eq. 17]:

$$\boxed{U_\text{RRR} = W_\text{LS} V_r \in \mathbb{R}^{m \times r}, \qquad V_\text{RRR} = V_r \in \mathbb{R}^{n \times r}, \qquad W_\text{RRR} = U_\text{RRR} V_\text{RRR}^\top}$$

Note that $V_\text{RRR}^\top V_\text{RRR} = I_r$ (semi-orthogonal) by construction, since $V_r$ collects orthonormal eigenvectors.

### 6.3 Derivation (Rank-1 Case)

To see why this algorithm is correct, work through the rank-1 case [eqs. 18--25]. The loss is:

$$\mathcal{L}(u, v) = \|Y - X u v^\top\|_F^2, \qquad \|v\| = 1$$

**Differentiate w.r.t.** $u$ holding $v$ fixed:

$$\frac{\partial \mathcal{L}}{\partial u} = -2 X^\top (Y - X u v^\top) v = 0 \implies X^\top X u = X^\top Y v \implies \hat{u} = (X^\top X)^{-1} X^\top Y v = W_\text{LS} v$$

**Substitute** $\hat{u} = W_\text{LS} v$ back into the loss and differentiate w.r.t. $v$ under constraint $\|v\|=1$:

$$\mathcal{L}(v) = \|Y\|_F^2 - v^\top W_\text{LS}^\top X^\top X W_\text{LS} v - \text{const}$$

Minimizing $\mathcal{L}$ w.r.t. $v$ is equivalent to **maximizing** $v^\top (W_\text{LS}^\top X^\top X W_\text{LS}) v$ subject to $\|v\|=1$. By the Rayleigh quotient theorem, the solution is:

$$\hat{v} = \text{top eigenvector of } M := W_\text{LS}^\top X^\top X W_\text{LS}$$

This is also the top *right* singular vector of $XW_\text{LS}$ (since the right singular vectors of $XW_\text{LS}$ are the eigenvectors of $(XW_\text{LS})^\top(XW_\text{LS}) = W_\text{LS}^\top X^\top X W_\text{LS} = M$).

For rank $r > 1$, the same argument extends by induction (each successive $v_k$ is the next eigenvector of $M$, orthogonal to the previous ones), giving Step 2 of the algorithm.

### 6.4 Why $Y^\top X W_\text{LS} = W_\text{LS}^\top X^\top X W_\text{LS}$

The matrix $M$ appearing in Step 2 can be written equivalently as $Y^\top X W_\text{LS}$. The algebraic identity is:

$$Y^\top X W_\text{LS} = Y^\top X (X^\top X)^{-1} X^\top Y = W_\text{LS}^\top X^\top X W_\text{LS}$$

The first equality substitutes the definition of $W_\text{LS}$; the second rearranges using $W_\text{LS}^\top = Y^\top X (X^\top X)^{-1}$.

**Consequence**: the top $r$ right singular vectors of $Y^\top X W_\text{LS}$ give $V_r$ directly. In practice:

- **R**: `svd(t(Y) %*% X %*% W_ls)$v[, 1:rank]`
- **Python**: `np.linalg.svd(Y.T @ X @ W_ls)[2][:rank, :].T`

Both give the same $V_r$ as the eigendecomposition of $M$, but the SVD route is numerically preferred when $M$ may have near-zero eigenvalues.

------------------------------------------------------------------------

## 7. Why RRR $\neq$ Low-Rank Approximation to $W_\text{LS}$

### 7.1 What SVD of $W_\text{LS}$ Minimizes

The Eckart-Young theorem says the rank-$r$ SVD of $W_\text{LS}$ solves:

$$\hat{W}_r^\text{SVD} = \arg\min_{\text{rank}(W) \leq r} \|W_\text{LS} - W\|_F^2$$

This minimizes the error in the *weight matrix*, treating all $m$ input directions equally.

RRR minimizes a different objective:

$$W_\text{RRR} = \arg\min_{\text{rank}(W) \leq r} \|Y - XW\|_F^2$$

This minimizes the error in the *predictions*, weighting input directions by their variance (through the factor $X^\top X$).

### 7.2 When They Agree

If $X^\top X = a I_m$ for some scalar $a > 0$ (spherical input distribution), then:

$$W_\text{LS} = \frac{1}{a} X^\top Y$$

The matrix $M = W_\text{LS}^\top X^\top X W_\text{LS} = a W_\text{LS}^\top W_\text{LS}$, whose eigenvectors are the right singular vectors of $W_\text{LS}$.

Therefore, in the spherical case, $V_r$ from Step 2 of RRR equals the top $r$ right singular vectors of $W_\text{LS}$, which is exactly what the SVD of $W_\text{LS}$ gives. The two methods agree.

They disagree whenever $X^\top X \not\propto I$: high-variance input directions are amplified in $M$ relative to the SVD of $W_\text{LS}$.

### 7.3 Geometric Intuition

Think of $XW_\text{LS}$ as the fitted output trajectory living in $\mathbb{R}^n$. The total variance in these predictions is:

$$\text{Var}(XW_\text{LS}) = \text{Tr}[W_\text{LS}^\top X^\top X W_\text{LS}]$$

RRR picks the rank-$r$ subspace (spanned by $V_r$) that captures the most variance in $XW_\text{LS}$. This is PCA of the fitted output trajectory. The PCA directions are weighted by $X^\top X$, so high-variance input directions contribute more.

SVD of $W_\text{LS}$ instead picks the subspace of $\mathbb{R}^n$ that captures the most "energy" of the weight matrix itself, regardless of how often different input directions occur.

**Practical consequence**: when one input neuron fires much more than another (non-spherical $X^\top X$), SVD of $W_\text{LS}$ may select output directions driven by the quieter neuron's weights (which could be large), while RRR selects output directions driven by the more active neuron's signal.

------------------------------------------------------------------------

## 8. Ridge-RRR

### 8.1 Motivation

When $T$ is small relative to $m$, $X^\top X$ is near-singular and $W_\text{LS}$ has large variance. Ridge regression adds a penalty on $\|W\|_F^2$ to shrink the weights:

$$\min_W \|Y - XW\|_F^2 + \lambda \|W\|_F^2$$

### 8.2 Ridge Estimate

Differentiating and setting to zero:

$$\boxed{W_\text{ridge} = (X^\top X + \lambda I_m)^{-1} X^\top Y} \tag{Appendix C.1}$$

The matrix $X^\top X + \lambda I_m$ is always invertible for $\lambda > 0$, even when $T < m$.

**Ridge-RRR** simply replaces $W_\text{LS}$ with $W_\text{ridge}$ in Steps 2--3 of the standard algorithm:

$$V_r = \text{top } r \text{ eigenvectors of } W_\text{ridge}^\top X^\top X W_\text{ridge}$$
$$W_\text{ridge-RRR} = W_\text{ridge} V_r V_r^\top$$

### 8.3 Ridge Penalty Simplification

Under the low-rank constraint $W = UV^\top$ with $V^\top V = I_r$:

$$\|W\|_F^2 = \|UV^\top\|_F^2 = \text{Tr}[V U^\top U V^\top] = \text{Tr}[U^\top U V V^\top] = \text{Tr}[U^\top U] = \|U\|_F^2$$

The last step uses $V V^\top \neq I$ in general, but $\text{Tr}[V U^\top U V^\top] = \text{Tr}[U^\top (V V^\top) U]$. However, when $V^\top V = I_r$:

$$\text{Tr}[V U^\top U V^\top] = \text{Tr}[U^\top U V^\top V] = \text{Tr}[U^\top U I_r] = \|U\|_F^2$$

(using the cyclic property of trace). So the ridge penalty $\lambda \|W\|_F^2 = \lambda \|U\|_F^2$ penalizes *only the input axes* $U$, not the output axes $V$ (which are constrained to be semi-orthogonal anyway). Intuitively, the regularization shrinks the magnitude of the communication signal, not the direction it is read out.

### 8.4 $\lambda$ Selection

Cross-validation: hold out rows of $(X, Y)$, fit $W_\text{ridge}$ on training rows, evaluate $R^2$ on held-out rows. `rrr_cv_rank` sweeps over all combinations of `max_rank` and `lambda_grid` simultaneously, returning the $(r^*, \lambda^*)$ pair with the highest mean held-out $R^2$.

**Practical guidance**: start with a log-spaced grid from $10^{-2}$ to $10^4$ times the mean eigenvalue of $X^\top X$; the optimal $\lambda$ tends to scale with the signal-to-noise ratio.

------------------------------------------------------------------------

## 9. Full-Covariance RRR

### 9.1 Motivation

Standard RRR assumes $\varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_n)$: all output neurons have equal, independent noise. Real neural data violates both assumptions:

- *Unequal noise*: neurons with higher firing rates tend to have larger variance.
- *Correlated noise*: shared fluctuations (e.g., up/down states) create correlated noise across neurons.

Ignoring this structure biases the estimated communication subspace toward high-noise output neurons.

### 9.2 Estimator

Relax to $\varepsilon_t \sim \mathcal{N}(0, \Sigma)$ [eq. 34]. The log-likelihood loss becomes:

$$\mathcal{L} = \text{Tr}\left[(Y - XUV^\top) \Sigma^{-1} (Y - XUV^\top)^\top\right] \tag{eq. 35}$$

**Closed-form solution given $\Sigma$** [eqs. 36--37]:

Let $\Sigma^{1/2}$ denote the symmetric matrix square root. Define the whitened matrix:

$$\tilde{M} = \Sigma^{-1/2} W_\text{LS}^\top X^\top X W_\text{LS} \Sigma^{-1/2}$$

Compute top $r$ eigenvectors of $\tilde{M}$, collecting them in $\tilde{V}_r$. Then:

$$\boxed{V_\text{fcRRR} = \Sigma^{1/2} \tilde{V}_r} \tag{eq. 36}$$

$$\boxed{U_\text{fcRRR} = W_\text{LS} \Sigma^{-1} V_\text{fcRRR}} \tag{eq. 37}$$

**Intuition**: $\Sigma^{-1/2}$ whitens the output space so that all output directions are equally weighted before finding the communication subspace. High-noise output directions are down-weighted. The result $V_\text{fcRRR} = \Sigma^{1/2} \tilde{V}_r$ maps back from the whitened space to the original output space.

### 9.3 Iterative Estimation of $\Sigma$

$\Sigma$ is unknown in practice. A coordinate-ascent algorithm alternates between updating $W$ and $\Sigma$ [coordinate ascent on the likelihood]:

1. Initialize $\Sigma = I_n$
2. Compute $W_\text{fcRRR}$ given current $\Sigma$ using eqs. 36--37
3. Update $\Sigma = \text{cov}(Y - X W_\text{fcRRR})$ (residual covariance)
4. Repeat steps 2--3 until $\max|W_\text{new} - W_\text{old}| < \text{tol}$ (typically converges in fewer than 10 iterations)

**Practical note**: at each iteration, $\Sigma$ must be positive definite. If the residual covariance is near-singular (which can happen when $n > T - mr$), add a small diagonal regularizer: $\Sigma \leftarrow \Sigma + \delta I$.

### 9.4 When to Use

- When output neurons have clearly unequal noise (e.g., different mean firing rates across neurons).
- When noise is correlated across neurons (e.g., shared slow fluctuations, common inputs).
- When you want the communication subspace to explain the *signal* in output activity, not the high-variance noise dimensions.
- Reduces exactly to standard RRR when $\Sigma = \sigma^2 I_n$.

------------------------------------------------------------------------

## 10. Comparison with Related Methods

| Method | Loss function | Description |
|---------------------|----------------------------------------------|---------------------------------------|
| RRR | $\|Y - XUV^\top\|_F^2$ | minimize prediction error, rank $r$ |
| SVD of $W_\text{LS}$ | $\|W_\text{LS} - UV^\top\|_F^2$ | best rank-$r$ approximation to weights |
| PCR | $\|Y - X \hat{U}_\text{PCA} V^\top\|_F^2$ | regress on top $r$ PCs of $X$ |
| CCA | $1 - \text{corr}(Yv, Xu)$ | maximize correlation, ignores variance |
| Ridge-RRR | $\|Y - XUV^\top\|_F^2 + \lambda\|U\|_F^2$ | RRR with shrinkage on input axes |
| fcRRR | $\text{Tr}[(Y-XUV^\top)\Sigma^{-1}(Y-XUV^\top)^\top]$ | RRR with non-spherical noise |

### 10.1 Relationship to PCR

Principal Components Regression (PCR) first decomposes $X = U_X S_X V_X^\top$ and retains only the top $r$ components of $X$, giving the reduced predictor $\tilde{X} = X V_{X,r} V_{X,r}^\top$. It then regresses $Y$ on $\tilde{X}$.

**Key assumption**: PCR assumes the most variable input dimensions (large eigenvalues of $X^\top X$) are also the most predictive of $Y$. This is sensible for noise-reduction but may miss low-variance input dimensions that have large effects on output.

**RRR makes no such assumption**: it finds the input directions that most reduce prediction error in $Y$, regardless of their variance in $X$. If a low-variance input dimension strongly drives the output, RRR will find it; PCR will not.

**When they agree**: if input activity and communication are both dominated by the same top-$r$ directions (i.e., $\alpha_\text{in} \approx 1$), PCR and RRR give nearly identical results.

### 10.2 Relationship to CCA

Canonical Correlation Analysis seeks pairs $(u, v)$ maximizing $\text{corr}(Xu, Yv)$. Unlike RRR, CCA normalizes both $Xu$ and $Yv$ to unit variance, so it is invariant to rescaling of $X$ or $Y$.

**Consequence**: CCA can identify a highly correlated but very low-variance communication channel that accounts for almost no output variance. RRR, by contrast, always selects the high-variance communication channel (since it maximizes $v^\top W_\text{LS}^\top X^\top X W_\text{LS} v$).

**When CCA is preferred**: when you want to find any correlated subspace regardless of variance (e.g., detecting a weak but reliable signal in a noisy population).

**When RRR is preferred**: when you want to explain *how much* of the output's variance is driven by the input (communication fraction, $R^2$).

------------------------------------------------------------------------

## 11. Alignment Metrics

### 11.1 Motivation

After fitting $W_\text{RRR}$, two natural questions arise:

1. **Input alignment**: are the input axes $U$ aligned with the dominant modes of input activity? Or does communication ride on the small input PCs?
2. **Output alignment**: does the communicated signal land on the dominant modes of output activity? Or does it drive the small output PCs?

These questions are answered by the input alignment index $\alpha_\text{in}$ and output alignment index $\alpha_\text{out}$, both in $[0, 1]$.

### 11.2 Communication Variance

The total variance in the communicated signal $XW = XUV^\top$ is:

$$\text{CommVar}(W) = \text{Tr}[W^\top \Sigma_X W], \qquad \Sigma_X = \frac{1}{T} X^\top X \tag{eq. 38}$$

This equals the sum of variances of each output neuron's predicted activity under $W$.

### 11.3 Input Alignment Index

Decompose $\Sigma_X$ in its eigenbasis: eigenvalues $\sigma_1^2 \geq \sigma_2^2 \geq \cdots \geq \sigma_m^2$ (input PC variances). Singular values of $W$: $\lambda_1 \geq \lambda_2 \geq \cdots \geq 0$ (padded with zeros to length $m$).

The communication variance is $\alpha_\text{in}^\text{raw} = \text{Tr}[W^\top \Sigma_X W]$ [eq. 38]. Its maximum over all rotations of $W$ with fixed singular values is achieved by pairing the largest $\lambda_k^2$ with the largest $\sigma_k^2$ (rearrangement inequality):

$$\alpha_\text{in}^\text{max} = \sum_{i=1}^{m} \lambda_i^2 \sigma_i^2 \tag{eq. 39}$$

$$\alpha_\text{in}^\text{min} = \sum_{i=1}^{m} \lambda_i^2 \sigma_{m+1-i}^2 \tag{eq. 40}$$

$$\boxed{\alpha_\text{in} = \frac{\alpha_\text{in}^\text{raw} - \alpha_\text{in}^\text{min}}{\alpha_\text{in}^\text{max} - \alpha_\text{in}^\text{min}}} \tag{eq. 41}$$

- $\alpha_\text{in} = 1$: communication is perfectly aligned with the top input PCs (PCR would work equally well).
- $\alpha_\text{in} = 0$: communication is anti-aligned (small input modes drive the output).
- $\alpha_\text{in} \approx 0.5$: communication is uniformly spread across input modes.

### 11.4 Communication Fraction

The **communication fraction** is the proportion of total output variance accounted for by the communicated signal:

$$\boxed{CF = \frac{\text{Tr}[W^\top \Sigma_X W]}{\text{Tr}[\Sigma_Y]}} \tag{eq. 42}$$

On training data, $CF \in [0, 1]$. On held-out data, $CF$ may exceed 1 if the model is misspecified or overfit.

### 11.5 Output Alignment Index

Decompose the output covariance: $\Sigma_Y = \sum_j \sigma_j^2 \mu_j \mu_j^\top$ (PCA of output). The variance communicated along output mode $j$ is:

$$\gamma_j^2 = \mu_j^\top (W^\top \Sigma_X W) \mu_j \tag{eq. 43}$$

The raw output alignment statistic is:

$$\alpha_\text{out}^\text{raw} = \sum_j \gamma_j^2 \sigma_j^2 \tag{eq. 44}$$

Normalize by computing the maximum (pair largest $\gamma_j^2$ with largest $\sigma_j^2$) and minimum (reverse pairing) over all permutations of $\{\gamma_j^2\}$ that preserve $\sum_j \gamma_j^2$ [eqs. 45--46]:

$$\boxed{\alpha_\text{out} = \frac{\alpha_\text{out}^\text{raw} - \alpha_\text{out}^\text{min}}{\alpha_\text{out}^\text{max} - \alpha_\text{out}^\text{min}}} \tag{eq. 47}$$

### 11.6 Key Difference from Input Alignment

For the input alignment index, the bound is achieved by rotating the input axes $U$ (which only changes $\alpha_\text{in}^\text{raw}$, not $\Sigma_X$). This rotation is freely available.

For the output alignment index, rotating $V$ would also change $\Sigma_Y$ (the output covariance), making the normalization bounds circular. Instead, the bound is computed by permuting $\{\gamma_j^2\}$ with $\{\sigma_j^2\}$ fixed — a combinatorial rearrangement, not a continuous rotation.

**Code note**: `main_figures.m` (fig. 6) uses an older formula for $\alpha_\text{out}^\text{raw}$ based on the difference of cumulative variance distributions (`muscom - muspop`). `alignment_output.m` and `alignment.py` implement the paper's eq. 44. `rrr_alignment_output` in R uses the paper formula.

------------------------------------------------------------------------

## 12. Practical Considerations

### 12.1 Centering

Both $X$ and $Y$ must be centered (subtract column means) before passing to `rrr_fit`. Without centering, the population mean acts as a constant communication dimension: $W$ will use a degree of rank to account for the mean offset, inflating the apparent communication rank by 1.

Centering is the *caller's responsibility*: `rrr_fit` receives pre-centered matrices.

### 12.2 Rank Selection

Cross-validation (via `rrr_cv_rank`) is the primary method. On held-out data, $R^2$ increases with $r$ initially, then decreases when the model overfits. The peak identifies $r^*$.

In small-$T$ regimes, this peak may be spuriously low (rank-1 is often "safe" even when the true rank is higher). Ridge regularization broadens the $R^2$ curve and allows higher ranks to be selected without overfitting (see Fig. 3 of Wu & Pillow 2025). Use `rrr_cv_rank` with a non-trivial `lambda_grid` to jointly select $r^*$ and $\lambda^*$.

### 12.3 Time Binning

For spike trains: the bin size should be at most as long as the temporal correlation of the fluctuations you want to capture. Bins that are too wide average out fast fluctuations, reducing effective $T$ and the signal in $X$.

For calcium imaging: the bin size is set by the calcium transient timescale (typically 200--500 ms). Using finer bins that are sub-Nyquist for the calcium signal adds correlated noise rows and inflates $T$ without adding information.

------------------------------------------------------------------------

## 13. Implementation Notes (R)

### 13.1 SVD Convention

R and Python differ in how SVD output is organized:

- **R**: `svd(M)$v` returns $V$ as a matrix with columns = right singular vectors. The decomposition is $M = U D V^\top$.
- **Python**: `np.linalg.svd(M)` returns $V_h = V^\top$ (rows = right singular vectors); `np.linalg.svd(M, full_matrices=False)[2]` is $V^\top$.

For the Step 2 computation with $M = Y^\top X W_\text{LS}$:

- **R**: `svd(M)$v[, 1:rank]` gives $V_r$ directly.
- **Python**: `np.linalg.svd(M)[2][:rank, :].T` gives the same $V_r$.

Both give identical columns (up to sign flips), because right singular vectors of $M$ are eigenvectors of $M^\top M = W_\text{LS}^\top X^\top X W_\text{LS}$.

### 13.2 Matrix Square Root

R has no built-in symmetric matrix square root for dense matrices. `expm::sqrtm` exists but is slow. For PSD matrices (such as $\Sigma$), use eigendecomposition:

$$\Sigma = Q D Q^\top \implies \Sigma^{1/2} = Q D^{1/2} Q^\top, \quad \Sigma^{-1/2} = Q D^{-1/2} Q^\top$$

`RRR_lib.R` implements `.mat_sqrt_psd(C)` and `.mat_inv_sqrt_psd(C)` via this approach. For near-singular $\Sigma$, eigenvalues below a tolerance are clamped to zero before inversion.

### 13.3 Condition Number Check in `rrr_fit`

```r
XX <- t(X) %*% X + lambda * diag(ncol(X))
if (rcond(XX) < 1e-10) {
  warning("ill-conditioned XX; using pseudoinverse")
  W_ls <- MASS::ginv(XX) %*% (t(X) %*% Y)
} else {
  W_ls <- solve(XX, t(X) %*% Y)
}
```

`rcond(XX) < 1e-10` indicates near-singularity. The `solve` path is faster and more accurate when the matrix is well-conditioned; `MASS::ginv` is a fallback.

### 13.4 `eigen` vs. `svd` for Step 2

For the PSD matrix $M = W_\text{LS}^\top X^\top X W_\text{LS}$, use `eigen(M)` (returns eigenvalues in *decreasing* order in R, so `$vectors[, 1:rank]` gives the top eigenvectors). Verify: `all(diff(eigen(M)$values) <= 0)` should be `TRUE`.

Alternatively, compute the SVD of $Y^\top X W_\text{LS}$ and take `$v[, 1:rank]`. This is numerically preferred when $M$ has eigenvalues near zero (SVD is more stable for rank-deficient matrices).

------------------------------------------------------------------------

## 14. Toy Example Walkthrough

$T = 6$ samples, $m = 2$ input neurons, $n = 2$ output neurons, true rank $r = 1$.

**True parameters**:

$$U = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad V = \begin{bmatrix} 0.8 \\ 0.6 \end{bmatrix}, \quad W = UV^\top = \begin{bmatrix} 0.8 & 0.6 \\ 0 & 0 \end{bmatrix}$$

Only input neuron 1 communicates; both output neurons receive, with weights $0.8$ and $0.6$.

**Input data** (pre-centered; alternating which neuron is active):

$$X = \begin{bmatrix} 1 & 0 \\ 2 & 0 \\ 3 & 0 \\ 0 & 1 \\ 0 & 2 \\ 0 & 3 \end{bmatrix}$$

**Noiseless output** $Y = XW$:

$$Y = \begin{bmatrix} 0.8 & 0.6 \\ 1.6 & 1.2 \\ 2.4 & 1.8 \\ 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}$$

**Step 1** — Compute $W_\text{LS}$:

$$X^\top X = \begin{bmatrix} 14 & 0 \\ 0 & 14 \end{bmatrix}, \qquad X^\top Y = \begin{bmatrix} 11.2 & 8.4 \\ 0 & 0 \end{bmatrix}$$

$$W_\text{LS} = (X^\top X)^{-1} X^\top Y = \frac{1}{14} \begin{bmatrix} 11.2 & 8.4 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.8 & 0.6 \\ 0 & 0 \end{bmatrix} = W \quad \checkmark$$

(Noiseless data: OLS recovers the true $W$ exactly.)

**Step 2** — Compute $M = W_\text{LS}^\top X^\top X W_\text{LS}$:

$$M = \begin{bmatrix} 0.8 & 0 \\ 0.6 & 0 \end{bmatrix} \begin{bmatrix} 14 & 0 \\ 0 & 14 \end{bmatrix} \begin{bmatrix} 0.8 & 0.6 \\ 0 & 0 \end{bmatrix} = 14 \begin{bmatrix} 0.64 & 0.48 \\ 0.48 & 0.36 \end{bmatrix}$$

This rank-1 matrix has one nonzero eigenvalue $14(0.64 + 0.36) = 14$ with eigenvector $\hat{v} = [0.8, 0.6]^\top$ (the true $V$).

**Step 3** — Compute $W_\text{RRR}$:

$$W_\text{RRR} = W_\text{LS} \hat{v} \hat{v}^\top = \begin{bmatrix} 0.8 & 0.6 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 0.64 & 0.48 \\ 0.48 & 0.36 \end{bmatrix} = \begin{bmatrix} 0.8 & 0.6 \\ 0 & 0 \end{bmatrix} = W \quad \checkmark$$

(Since $W_\text{LS}$ already has rank 1 with right singular vector $[0.8, 0.6]^\top$, the rank-1 RRR projection is an identity operation.)

**Alignment check**: $X^\top X = 14 I_2$ (spherical input), so $\alpha_\text{in}$ = 0.5 (communication uses one of two equal-variance directions). $CF = \text{Tr}[W^\top \Sigma_X W] / \text{Tr}[\Sigma_Y]$: the communication variance is $14 \cdot (0.8^2 + 0.6^2) \cdot 1 = 14$ and total output variance equals the same (noiseless), so $CF = 1$.

------------------------------------------------------------------------

## 15. Summary of Key Equations

$$\boxed{W_\text{LS} = (X^\top X)^{-1} X^\top Y} \tag{OLS}$$

$$\boxed{W_\text{ridge} = (X^\top X + \lambda I_m)^{-1} X^\top Y} \tag{Ridge}$$

$$\boxed{W_\text{RRR} = W_\text{LS} V_r V_r^\top, \quad V_r = \text{top-}r\text{ eigvecs of } W_\text{LS}^\top X^\top X W_\text{LS}} \tag{RRR}$$

$$\boxed{U_\text{RRR} = W_\text{LS} V_r, \quad V_\text{RRR} = V_r, \quad W_\text{RRR} = U_\text{RRR} V_\text{RRR}^\top} \tag{Factored}$$

$$\boxed{V_\text{fcRRR} = \Sigma^{1/2}\,\tilde{V}_r, \quad U_\text{fcRRR} = W_\text{LS} \Sigma^{-1} V_\text{fcRRR}} \tag{fcRRR}$$

$$\boxed{\alpha_\text{in} = \frac{\text{Tr}[W^\top \Sigma_X W] - \alpha_\text{in}^\text{min}}{\alpha_\text{in}^\text{max} - \alpha_\text{in}^\text{min}}} \tag{Input alignment}$$

$$\boxed{CF = \frac{\text{Tr}[W^\top \Sigma_X W]}{\text{Tr}[\Sigma_Y]}} \tag{Communication fraction}$$

$$\boxed{\alpha_\text{out} = \frac{\sum_j \gamma_j^2 \sigma_j^2 - \alpha_\text{out}^\text{min}}{\alpha_\text{out}^\text{max} - \alpha_\text{out}^\text{min}}, \quad \gamma_j^2 = \mu_j^\top (W^\top \Sigma_X W) \mu_j} \tag{Output alignment}$$
