# demixed PCA (dPCA) 数学ノート

**Reference**: Kobak, D. et al. (2016). Demixed principal component analysis of neural population data. *eLife* 5:e10989.  
**R 実装**: `demixedPCA_lib.R`（このプロジェクト）  
**視覚的補足**: [dPCA_anova_decomp.html](dPCA_anova_decomp.html)  
**jPCA との接続**: [jPCA_notes.md](jPCA_notes.md)

> このノートは dPCA の数学的理解のためのものです。実装コードは含みません。

---

## 1. なぜ PCA では不十分か

標準 PCA は「分散を最大化する軸」を見つける。問題は、その軸が複数のタスク変数を混ぜることです：

```
PC1 が高いとき、それは
  (a) rule が変わったから？
  (b) stim が変わったから？
  (c) 両方が変わったから？
  → PCA には区別する手段がない
```

**dPCA の目標**：各タスクパラメータ（time、stimulus、rule、...）に起因する分散を**分離**し、
それぞれを最もよく説明するコンポーネントを求める。

---

## 2. データ構造と記法

| 記号 | 意味 |
|------|------|
| $N$ | チャンネル数（EEG 電極など） |
| $K$ | タスクパラメータの数（軸の数） |
| $d_k$ | $k$ 番目のパラメータ軸のサイズ |
| $X \in \mathbb{R}^{N \times d_1 \times \cdots \times d_K}$ | 試行平均データ |
| $\phi \subseteq \{1,\ldots,K\}$ | パラメータ軸のサブセット（"marginalization"） |
| $X^{(\phi)}$ | $\phi$ に対応するマージナリゼーション |

**中心化**：全ニューロン（チャンネル）の全条件にわたる平均を引く：

$$\tilde{X}_{n,i_1,\ldots,i_K} = X_{n,i_1,\ldots,i_K} - \frac{1}{d_1 \cdots d_K} \sum_{j_1,\ldots,j_K} X_{n,j_1,\ldots,j_K}$$

---

## 3. マージナリゼーション（ANOVA 分解）

### 核心アイデア

$$\tilde{X} = \sum_{\phi \neq \emptyset} X^{(\phi)}$$

各 $X^{(\phi)}$ は「因子 $\phi$ だけに起因する変動」を含む純粋な成分。これは **2 要因 ANOVA の効果分解** と全く同じ考え方。

### 2 パラメータの例（$K=2$：labels = "st"）

データ：$\tilde{X}[N, S, T]$（$S$：stimulus、$T$：time）

**Step A：各パラメータの条件平均**

$$\bar{X}^{(s)}_{n,s} = \frac{1}{T}\sum_t \tilde{X}_{n,s,t} \qquad \text{(Tを平均、sが残る)}$$

$$\bar{X}^{(t)}_{n,t} = \frac{1}{S}\sum_s \tilde{X}_{n,s,t} \qquad \text{(Sを平均、tが残る)}$$

**Step B：主効果を取り出す（全体平均を引く）**

$$X^{(s)}_{n,s,t} = \bar{X}^{(s)}_{n,s} \qquad \text{(tによらない)}$$

$$X^{(t)}_{n,s,t} = \bar{X}^{(t)}_{n,t} \qquad \text{(sによらない)}$$

**Step C：交互作用を取り出す（主効果を引く）**

$$X^{(st)}_{n,s,t} = \tilde{X}_{n,s,t} - X^{(s)}_{n,s,t} - X^{(t)}_{n,s,t}$$

**検証**：$X^{(s)} + X^{(t)} + X^{(st)} = \tilde{X}$ ✓

ANOVA との対応：

| dPCA 項 | ANOVA 名称 | 解釈 |
|---------|-----------|------|
| $X^{(s)}$ | stimulus の主効果 | stimulus で変化するが time では変化しない |
| $X^{(t)}$ | time の主効果 | time で変化するが stimulus では変化しない |
| $X^{(st)}$ | 交互作用 | stimulus と time が同時に変化するときのみ現れる |

### $K$ パラメータの一般式（包除原理）

$$X^{(\phi)} = \operatorname{mean}_{\phi^c}(\tilde{X}) - \sum_{\psi \subsetneq \phi} X^{(\psi)}$$

- $\phi^c = \{1,\ldots,K\} \setminus \phi$：補集合（$\phi$ に含まれない軸）
- $\operatorname{mean}_{\phi^c}$：$\phi^c$ の軸方向に平均（$\phi$ の軸は保持）
- 小さいサブセットから順番に計算する（式の右辺に既計算の $X^{(\psi)}$ を使う）

**合計**: $2^K - 1$ 個のサブセット（空集合を除く）

3 パラメータ（"trs"）では 7 個：t, r, s, tr, ts, rs, trs。

### なぜマージナリゼーションが Frobenius 直交するか

$$\langle X^{(\phi)}, X^{(\psi)} \rangle_F = \sum_{n,i_1,\ldots,i_K} X^{(\phi)}_{n,i_1,\ldots,i_K} \cdot X^{(\psi)}_{n,i_1,\ldots,i_K} = 0 \quad (\phi \neq \psi)$$

$X^{(\phi)}$ は「$\phi$ 方向にのみ変化し、$\phi^c$ 方向の変化成分がゼロ」になるため、
異なる方向に特化した成分の内積は対称性からゼロになる（ANOVA の「効果の非交絡性」と同じ原理）。

**重要な帰結**：分散の分割が成立する。

$$\sum_\phi \|X^{(\phi)}\|_F^2 = \|\tilde{X}\|_F^2$$

---

## 4. 最適化問題

各マージナリゼーション $\phi$ に対して、**encoder** $F_\phi$ と **decoder** $D_\phi$（ともに $N \times k$ 行列）を求める：

$$\mathcal{L}_\phi(F_\phi, D_\phi) = \left\| X^{(\phi)} - F_\phi D_\phi^\top \tilde{X} \right\|_F^2 \to \min$$

全マージナリゼーションを合わせると：

$$\min_{F_\phi, D_\phi} \sum_\phi \left\| X^{(\phi)} - F_\phi D_\phi^\top \tilde{X} \right\|_F^2$$

**損失関数の意味**：「$\tilde{X}$ の $k$ 次元の部分空間への射影が、$\phi$ 成分を最もよく再現する」ような
encoder と decoder を見つける。

---

## 5. 閉形式解

### Bridge 行列 $C_\phi$

$$C_\phi = X^{(\phi)}_\text{flat} \cdot \left(\tilde{X}_\text{flat}\right)^\dagger \in \mathbb{R}^{N \times N}$$

ここで "flat" は $[N, d_1 \cdots d_K]$ に reshape、$^\dagger$ は Moore-Penrose 擬逆行列。

**$C_\phi$ の意味**：「マージナリゼーション $X^{(\phi)}$ が、全データ $\tilde{X}$ からどう線形写像されて生まれるか」。
$X^{(\phi)} = \tilde{X}$ のとき $C_\phi = I$ → dPCA が標準 PCA に一致する。

### 最適 $F_\phi$ と $D_\phi$

$C_\phi \tilde{X}_\text{flat}$ を SVD 分解する：

$$C_\phi \tilde{X}_\text{flat} = U_\phi \Sigma_\phi V_\phi^\top \quad \text{（上位 $k$ 成分に切り捨て）}$$

$$\boxed{F_\phi = U_\phi} \qquad \boxed{D_\phi = C_\phi^\top U_\phi}$$

### なぜ $F_\phi \neq D_\phi$（非対称性）

標準 PCA では encoder = decoder = 主軸。dPCA では異なる：

- **$D_\phi$（decoder）**：$\tilde{X}$ を射影する方向 → factor $\phi$ の分散を最大化
- **$F_\phi$（encoder）**：ニューラル空間に戻す方向 → $X^{(\phi)}$ を最適に再現

この非対称性が "demixed"（分離された）の核心で、潜在コード $Z_\phi = D_\phi^\top \tilde{X}$ が
他の因子の分散を含んでいても、$F_\phi Z_\phi$ は $X^{(\phi)}$ を最小二乗で再現する。

### 双直交性（Biorthogonality）

$$D_\phi^\top F_\phi \approx I_{k}$$

「$D_\phi$ で射影した潜在コードを $F_\phi$ で戻すと恒等変換に近い」→ 各潜在次元が 1 つの因子だけに
対応することを保証する。

---

## 6. Transform と Inverse Transform

**射影（transform）**：

$$Z_\phi = D_\phi^\top \tilde{X}_\text{flat} \in \mathbb{R}^{k \times (d_1 \cdots d_K)}$$

これを元の条件次元に reshape すると $[k \times d_1 \times \cdots \times d_K]$。

**逆変換（inverse transform）**：

$$\hat{X}^{(\phi)} = F_\phi Z_\phi = F_\phi D_\phi^\top \tilde{X}_\text{flat} \in \mathbb{R}^{N \times (d_1 \cdots d_K)}$$

---

## 7. Explained Variance

マージナリゼーション $\phi$、成分 $j$ の説明分散：

$$R^2_{\phi,j} = \frac{\|D_{\phi,j}^\top \tilde{X}_\text{flat}\|_2^2}{\|\tilde{X}_\text{flat}\|_F^2}$$

- 分子：第 $j$ 成分の射影の分散  
- 分母：全データの総分散

**注意**：異なるマージナリゼーション・成分の $R^2$ を合計すると 1.0 を超えることがある。
各 $D_{\phi,j}$ がグローバルに直交していないため（Frobenius 直交するのは $X^{(\phi)}$ であり $D_\phi$ ではない）。

---

## 8. 正則化

$N > d_1 \cdots d_K$ のとき（チャンネル数 > 条件数 × 時間数）、擬逆行列が ill-conditioned になる。
Tikhonov 正則化：

$$\tilde{X}^\text{reg}_\text{flat} = \left[\tilde{X}_\text{flat} \;\Big|\; \lambda I_N \right] \in \mathbb{R}^{N \times (d + N)}$$

各 $X^{(\phi)}_\text{flat}$ にも $\lambda$ 倍のゼロ列を append する。これは以下の正則化に相当：

$$\min_{F, D} \| X^{(\phi)}_\text{flat} - F D^\top \tilde{X}_\text{flat} \|_F^2 + \lambda^2 \|F\|_F^2$$

**EEG の典型的な設定**（$N = 150$, $d_1 \cdots d_K = T \times C = 250 \times 6 = 1500$）では
$N < K$ なので正則化は不要。

---

## 9. 有意性検定

各コンポーネントが有意かどうかを shuffle test で確認：

1. 試行データを train/test に分割
2. train で $D_\phi$ を fit
3. test の $Z_\phi = D_\phi^\top X_\text{test}$ に対して 1-NN 分類を適用
4. 条件ラベルをシャッフルして null 分布を作る
5. 真のスコア > null の最大値 → significant（$p < 1/n_\text{shuffles}$）
6. 連続した有意 timepoint の数が $n_\text{consecutive}$ 以上のみを有意とする（偶発的な単点を除外）

---

## 10. アルゴリズムのまとめ

```
Input:
  X_avg  [N × d_1 × ... × d_K]   試行平均データ
  labels  "trs"（パラメータ軸名の文字列）
  n_components: int または named list
  regularizer: 0（デフォルト）

──────────────────────────────────────────────
Step 1: ANOVA 分解（dpca_marginalize）

  μ = rowMeans(X_flat)   # 全体平均
  Xtilde_flat = X_flat - μ

  for each φ in get_marginalizations(labels):    # 2^K - 1 個
    X_phi = mean over φ^c axes - sum of X_psi (ψ ⊊ φ)
    X_phi_flat = matrix(X_phi, nrow=N)

──────────────────────────────────────────────
Step 2: Bridge 行列と SVD

  X_pinv = ginv(Xtilde_flat)   # Moore-Penrose 擬逆行列

  for each φ:
    C_phi = X_phi_flat %*% X_pinv           # [N × N]
    M = C_phi %*% Xtilde_flat               # [N × K]
    svd(M) → U, s, V
    P[[φ]] = U[, 1:k]                       # encoder = F_phi
    D[[φ]] = t(C_phi) %*% U[, 1:k]         # decoder = D_phi

──────────────────────────────────────────────
Step 3: Transform

  for each φ:
    Z[[φ]] = t(D[[φ]]) %*% Xtilde_flat     # [k × K]
    reshape to [k × d_1 × ... × d_K]

Output:
  P, D     : encoders, decoders（各因子の demixed components）
  Z        : projected data（各因子の潜在表現）
  var_exp  : explained variance ratio per marginalization
```

---

## 11. dPCA → jPCA への接続

dPCA で得られた $Z^{(t)}$（time marginalization の潜在表現）に対して jPCA をかける：

```
dpca_model <- dpca_fit(X, "ts", n_components = 6)
Z_t        <- dpca_transform(X, dpca_model)[["t"]]   # [6 × S × T] を条件ごとに分割
jpca_model <- jpca_fit(Z_t_list)
```

**意味**：「time に関する表現」は時間的に回転しているか？  
raw EEG に直接 jPCA をかけると、rotation が何の因子（stimulus？time？）によるものか判別できない。
dPCA で demix してから jPCA をかけることで、純粋な time-related dynamics の回転性を確認できる。

詳細は `ARCHITECTURE_demixed_j_PCA.md` の Phase 3 "dpca_jpca_pipeline" を参照。

---

## 12. 参考文献

- **Kobak, D. et al. (2016).** Demixed principal component analysis of neural population data. *eLife* 5:e10989.  
  dPCA のオリジナル論文。Methods section に全アルゴリズムの詳細。Fig.2：surrogate data でのデモ。
- **Brendel, W. et al. (2011).** Demixed principal component analysis. *NIPS 2011.*  
  前身論文。基本的なアイデアが先に提示されている。
- **machenslab/dPCA** (Python/MATLAB). https://github.com/machenslab/dPCA  
  Reference implementation。数値検証に使用。
