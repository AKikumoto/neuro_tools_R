# jPCA 数学ノート

**Reference**: Churchland, M.M. et al. (2012). Neural population dynamics during reaching. *Nature* 487, 51–56.  
**R 実装**: `jPCA_lib.R`（このプロジェクト）  
**視覚的補足**: [jPCA_geometry.html](jPCA_geometry.html)

> このノートは jPCA の数学的理解のためのものです。実装コードは含みません。
> dPCA と組み合わせて使う位置づけについては [demixedPCA_notes.md](demixedPCA_notes.md) を参照してください。

---

## 1. 問いの立て方

EEG デコーディングは「時刻 $t$ に rule が解読できるか」を問う。  
jPCA はそれとは別の問いを立てる：

> **Population state の変化の仕方**（dynamics）は回転的か？

Population state とは、$N$ チャンネルの活動を時刻 $t$ でまとめた $N$ 次元ベクトル：

$$\mathbf{x}(t) \in \mathbb{R}^N$$

時間とともに $N$ 次元空間内を動く「点の軌跡」= **trajectory** を観察する。

---

## 2. 線形ダイナミクスの仮定

最もシンプルな仮定：**状態の変化（velocity）が現在の状態に線形比例する**。

$$\dot{\mathbf{x}}(t) = M \mathbf{x}(t)$$

- $\dot{\mathbf{x}} = d\mathbf{x}/dt$（有限差分 $\mathbf{x}(t+1) - \mathbf{x}(t)$ で近似）  
- $M \in \mathbb{R}^{N \times N}$：推定したいダイナミクス行列

無制約の $M$ は回転・拡大・歪みを全部含む。「dynamics が回転的か」という問いに答えるには、$M$ を **skew-symmetric（歪対称）** に制約する。

---

## 3. Skew-symmetric 行列と回転

### 定義

$$M_\text{skew} = -M_\text{skew}^\top$$

具体例（$2 \times 2$）：

$$M_\text{skew} = \begin{pmatrix} 0 & -\omega \\ \omega & 0 \end{pmatrix}$$

$m_{ii} = 0$（対角はゼロ）、$m_{ij} = -m_{ji}$（上下三角が符号反転）。  
自由度は $N(N-1)/2$ — $N=2$ のとき 1 個（角速度 $\omega$ だけ）。

### なぜ「回転」を意味するか

$\dot{\mathbf{x}} = M_\text{skew}\,\mathbf{x}$ の解は：

$$\mathbf{x}(t) = e^{M_\text{skew}\,t}\,\mathbf{x}(0)$$

$M_\text{skew}$ が skew-symmetric のとき：

$$\bigl(e^{M_\text{skew}}\bigr)^\top = e^{M_\text{skew}^\top} = e^{-M_\text{skew}} = \bigl(e^{M_\text{skew}}\bigr)^{-1}$$

→ $e^{M_\text{skew}}$ は**直交行列**（$O^\top = O^{-1}$）= 回転行列（または鏡映）。

結論：$\|\mathbf{x}(t)\| = \|\mathbf{x}(0)\|$（大きさ不変）、方向だけが変化 = **等速円運動**。

### 固有値が純虚数になる証明

$M_\text{skew}\,\mathbf{v} = \lambda \mathbf{v}$ に左から $\bar{\mathbf{v}}^\top$ をかける：

$$\bar{\mathbf{v}}^\top M_\text{skew}\,\mathbf{v} = \lambda\,|\mathbf{v}|^2$$

この式の複素共役を取ると：

$$\overline{\bar{\mathbf{v}}^\top M_\text{skew}\,\mathbf{v}} = \mathbf{v}^\top M_\text{skew}^\top \bar{\mathbf{v}} = -\mathbf{v}^\top M_\text{skew}\,\bar{\mathbf{v}} = -\overline{\bar{\mathbf{v}}^\top M_\text{skew}\,\mathbf{v}}$$

→ $\bar{\mathbf{v}}^\top M_\text{skew}\,\mathbf{v}$ は純虚数（または 0）  
→ $\lambda\,|\mathbf{v}|^2$ が純虚数 → $\lambda = \pm i\omega$（$\omega \in \mathbb{R}$）。

固有値 $\lambda = i\omega$ のとき、解は Euler の公式より：

$$e^{i\omega t} = \cos(\omega t) + i\sin(\omega t)$$

→ 角周波数 $\omega$ でスピンする。

---

## 4. 角度 $\theta$ による回転強度の定量化

jPCA 平面に射影した後、各 (条件 $c$, 時刻 $t$) について：

$$\theta(t,c) = \operatorname{atan2}\!\bigl(\mathbf{x} \wedge \dot{\mathbf{x}},\; \mathbf{x} \cdot \dot{\mathbf{x}}\bigr)$$

ここで $\mathbf{x} \cdot \dot{\mathbf{x}}$ は内積、$\mathbf{x} \wedge \dot{\mathbf{x}} = x_1 \dot{x}_2 - x_2 \dot{x}_1$ は 2D 外積の $z$ 成分。

#### 内積と $\cos\theta$ の関係

$$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos\theta$$

（2 ベクトルの成分展開から加法定理を通じて自然に出てくる。）  
直観：$\mathbf{b}$ を $\mathbf{a}$ 方向に射影した長さ $\times |\mathbf{a}|$。

#### 外積 $z$ 成分と $\sin\theta$ の関係

$$a_1 b_2 - a_2 b_1 = |\mathbf{a}||\mathbf{b}|\sin\theta$$

直観：2 ベクトルが作る平行四辺形の**符号付き面積**。

#### atan2

$$\theta = \operatorname{atan2}(y,\,x)$$

- 第 1 引数 $y = \sin\theta$、第 2 引数 $x = \cos\theta$
- 返り値 $\in (-\pi, \pi]$ — 4 象限を一意に区別できる
- 通常の $\arctan(y/x)$ は商が同じなら同一象限として扱うため不十分

$|\mathbf{a}||\mathbf{b}|$ は分子・分母両方にかかるのでキャンセルされ：

$$\theta = \operatorname{atan2}(\mathbf{x} \wedge \dot{\mathbf{x}},\; \mathbf{x} \cdot \dot{\mathbf{x}})$$

#### $\theta$ の 4 ケース（視覚的補足 → [jPCA_geometry.html](jPCA_geometry.html)）

| $\theta$ | 幾何的意味 |
|---|---|
| $\approx +\pi/2$ | 純粋な反時計回り回転（$\dot{\mathbf{x}} \perp \mathbf{x}$） |
| $\approx 0$ | 純粋な拡大（$\dot{\mathbf{x}} \parallel \mathbf{x}$） |
| $\approx -\pi/2$ | 純粋な時計回り回転 |
| $\approx \pm\pi$ | 純粋な収縮（原点に向かう） |

$\theta$ 分布のピークが $\pi/2$ に集中 → rotational dynamics あり。

---

## 5. $R^2$ 比による回転強度の定量化

$\theta$ とは独立した別の指標：

$$R^2_\text{ratio} = \frac{R^2_\text{skew}}{R^2_\text{unrestr}}$$

- $R^2_\text{unrestr}$：無制約の $\hat{M}$ が $\dot{X}$ を説明できる割合（上限）
- $R^2_\text{skew}$：$M_\text{skew}$ が $\dot{X}$ を説明できる割合

$$R^2 = 1 - \frac{\|dX - M X_\text{prev}\|_F^2}{\|dX\|_F^2}$$

解釈の目安：

| $R^2_\text{ratio}$ | 意味 |
|---|---|
| $\approx 1.0$ | dynamics がほぼ純粋な回転 |
| $\approx 0.5$ | ランダムデータの期待値 |
| $\ll 0.5$ | 起こり得ない（$M_\text{skew}$ は $\hat{M}$ の成分の一つだから） |

ランダムデータで期待値が $0.5$ である理由：任意の行列 $\hat{M}$ は対称成分 $S$ と歪対称成分 $A$ に直交分解できる（Frobenius 内積がゼロ）。

$$\hat{M} = \underbrace{\frac{\hat{M} + \hat{M}^\top}{2}}_{S} + \underbrace{\frac{\hat{M} - \hat{M}^\top}{2}}_{A = M_\text{skew}}$$

ランダムデータでは $S$ と $A$ の説明力が平均的に等しくなるため、$R^2_\text{ratio} \approx 0.5$。

**注意**：normalize=FALSE（PC 間の分散スケールを揃えない）にすると $R^2_\text{ratio}$ が負になることがある。これは PCA の各 PC の分散が大きく異なるとき、M_skew の推定が歪むため。`jPCA_lib.R` では `normalize=TRUE` がデフォルト。

---

## 6. アルゴリズムのまとめ

```
Input: X_list — 条件ごとの [N × T] 行列のリスト（条件数 ≥ 3）

Step 1: PCA 前処理
  (a) 全条件を結合: X_full [N × C*T]
  (b) cross-condition mean を引く
      各 timepoint で全条件の平均を引く（共通 evoked response を除去）
  (c) PCA → top n_pcs PC を保持（典型: n_pcs = 6）
      → X_red [n_pcs × C*T]
  (d) 各 PC を PC 固有値の平方根（= 標準偏差）で割って正規化
      （PC 間のスケール差を除き、M_skew の推定を安定化）

Step 2: 有限差分
  dX     = X[t+1] − X[t]   [n_pcs × C*(T-1)]
  X_prev = X[1:(T-1)]      [n_pcs × C*(T-1)]

Step 3: 無制約 M を最小二乗推定
  dX ≈ M X_prev
  M_hat = dX X_prev^T (X_prev X_prev^T)^{-1}

Step 4: Skew-symmetric に射影
  M_skew = (M_hat − M_hat^T) / 2
  （任意の行列の歪対称成分 = Frobenius ノルム最小の skew-symmetric 近似）

Step 5: 固有値分解
  eigen(M_skew) → λ = ±iω, V（複素共役ペア）
  |ω| の大きい順に並べ替え

Step 6: 実数の回転平面を復元
  複素共役ペア (v₁, v̄₁) から：
    jPC1 = Re(v₁)（実部をそのままとり正規化）
    jPC2 = −Im(v₁)（虚部の符号を反転して正規化）
  これらは直交し、複素固有空間と同じ部分空間を張る

Step 7: データを射影
  W = [jPC1; jPC2]  [2 × n_pcs]
  X_jPCA = W X_red  [2 × C*T]

Output:
  W         : jPC axes（2 × n_pcs）
  M_skew    : fitted skew-symmetric dynamics matrix
  R²_skew   : R² of M_skew fit
  R²_unrestr: R² of unconstrained M fit
  eig_freq  : eigenvalue magnitudes |ω| for each jPC pair
```

### jPC1 と jPC2 がなぜ直交するか

$\mathbf{v}_1 = \mathbf{a} + i\mathbf{b}$（$\mathbf{a} = \text{Re}(\mathbf{v}_1)$, $\mathbf{b} = \text{Im}(\mathbf{v}_1)$）とおくと：

$$\text{jPC1} \cdot \text{jPC2} = \mathbf{a} \cdot (-\mathbf{b})$$

$M_\text{skew}$ は実 skew-symmetric なので固有値は純虚数で、複素固有ベクトルの実部と虚部が直交することが示せる（$\mathbf{a}^\top \mathbf{b} = 0$）。

### cross-condition mean を引く理由

全条件に共通する強い evoked response（例：刺激提示直後の共通反応）があると、$\hat{M}$ はその時間的ドリフトを「回転」として誤検出する。条件間の差分だけを残すことで、純粋な条件依存構造が見える。

---

## 7. 疑問チェック

1. $M_\text{skew}$ の固有値がなぜ純虚数か？（→ Section 3 の証明）
2. jPC1 ⊥ jPC2 の理由は？（→ Section 6 末尾）
3. $R^2_\text{ratio} \approx 0.5$ がランダムの期待値な理由は？（→ Section 5）
4. normalize=FALSE で $R^2_\text{ratio}$ が負になる理由は？（→ Section 5 注意）
5. 条件数が 3 以上必要な理由は？（binary では jPCA 平面が 1 次元になり回転を定義できない）

---

## 8. 参考文献

- **Churchland, M.M. et al. (2012).** Neural population dynamics during reaching. *Nature* 487, 51–56.  
  jPCA のオリジナル論文。Fig.3: 回転軌跡の可視化、Fig.6: $\theta$ 分布。Supplementary Methods にアルゴリズム詳細。
