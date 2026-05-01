# jPCA テスト結果の説明

**テストスクリプト**: `test/test_jpca_fit.R`（10 テスト、全 PASS）  
**実行方法**: `Rscript -e "library(testthat); source('jPCA_lib.R'); source('test/test_jpca_fit.R')"`  
**関連ノート**: [jPCA_notes.md](jPCA_notes.md)

> このドキュメントはテストスクリプトの結果を説明するものです。各テストが何を確認しているか、
> なぜその条件が成り立つかを記述します。実装コードは含みません。

---

## サーロゲートデータ

### `make_rotation_data`

純粋な 2D 回転を $N$ 次元に埋め込んだデータ：

- 2D 平面で $\mathbf{z}(t) = (\cos(\omega t + \theta_0),\, \sin(\omega t + \theta_0))$
- ランダム混合行列 $A \in \mathbb{R}^{N \times 2}$ で $N$ 次元に射影：$\mathbf{x}^{(c)}(t) = A\,\mathbf{z}(t) + \varepsilon$
- 条件 $c$ ごとに初期角度 $\theta_0 = 2\pi(c-1)/C$ を均等配置
- $\varepsilon$：ガウスノイズ（`noise_sd=0.05`）— フルランクにするため必要

パラメータ：`N=6, T=50, C=4, omega=0.2, noise_sd=0.05, seed=42`

### `make_random_data`

回転構造を持たない純粋なホワイトノイズ：`N=6, T=50, C=4`

---

## Test 1 — M_skew が歪対称

**確認内容**: `max(|M_skew + M_skew^T|) < 1e-10`

**なぜ成り立つか**:  
`jpca_fit` の Step 4 で $M_\text{skew} = (\hat{M} - \hat{M}^\top)/2$ と定義している。
定義から $M_\text{skew} + M_\text{skew}^\top = (\hat{M} - \hat{M}^\top)/2 + (\hat{M}^\top - \hat{M})/2 = 0$。
数値誤差は $O(\epsilon_\text{machine}) \approx 10^{-16}$ のオーダーで、1e-10 を大きく下回る。

---

## Test 2 — M_skew の固有値が純虚数

**確認内容**: `max(|Re(eigenvalues)|) < 1e-10`

**なぜ成り立つか**:  
Skew-symmetric 行列の固有値は必ず純虚数 $\lambda = \pm i\omega$（$\omega \in \mathbb{R}$）になる。証明は [jPCA_notes.md Section 3](jPCA_notes.md)。  
実際には浮動小数点演算により $\text{Re}(\lambda)$ は厳密にゼロではなく $O(10^{-15})$ になるが、
閾値 1e-10 内に収まる。

---

## Test 3 — jPC1 ⊥ jPC2

**確認内容**: `|W[1,] · W[2,]| < 1e-10`

**なぜ成り立つか**:  
jPC1 = $\text{Re}(\mathbf{v}_1)$、jPC2 = $-\text{Im}(\mathbf{v}_1)$ と定義する（$\mathbf{v}_1$ は $M_\text{skew}$ の固有ベクトル）。
実 skew-symmetric 行列の複素固有ベクトルでは $\text{Re}(\mathbf{v}_1) \perp \text{Im}(\mathbf{v}_1)$ が成り立つ
（[jPCA_notes.md Section 6](jPCA_notes.md)参照）。  
さらに各 jPC は単位ベクトルに正規化されるので、内積はほぼ 0 になる。

---

## Test 4 — jPC1 と jPC2 が単位ベクトル

**確認内容**: `||W[1,]|| = 1` かつ `||W[2,]|| = 1`（tolerance 1e-10）

**なぜ成り立つか**:  
`jpca_fit` の Step 6 で各 jPC を `v / norm(v)` と明示的に正規化している。
浮動小数点誤差は $O(10^{-15})$ で閾値 1e-10 内。

---

## Test 5 — $R^2_\text{skew} \leq R^2_\text{unrestr}$

**確認内容**: `R2_skew <= R2_unrestr + 1e-10`

**なぜ成り立つか**:  
$M_\text{skew}$ は無制約 $\hat{M}$ の探索空間（全 $N \times N$ 行列）の部分集合（歪対称行列のみ）に
制約している。制約を加えれば最小二乗の残差は増えるかゼロのまま → $R^2$ は減るかゼロのまま。  
数値誤差用に +1e-10 の余裕を設けている。

---

## Test 6 — 純粋回転データで $R^2_\text{ratio} > 0.85$

**確認内容**: `R2_ratio > 0.85`

**何を確認しているか**:  
データが純粋な回転から生成されているため、skew-symmetric 制約を付けても説明力がほとんど
落ちないことを確認する。`R2_ratio = R2_skew / R2_unrestr` が 0.85 を超えることで、
「dynamics が高度に回転的である」と言える。  

数値的な詳細：`noise_sd=0.05` は小さいが完全にゼロではないため、$R^2_\text{ratio}$ は
理論値 1.0 から少し下がる。0.85 という閾値は、ノイズの影響を許容しつつ回転性を確認できる値。

---

## Test 7 — ピーク角度が π/2 の近傍

**確認内容**: `|peak - π/2| < 0.3`

**何を確認しているか**:  
`jpca_rotation_strength()` が全 (条件, 時刻) の $\theta$ ヒストグラムのピーク位置を返す。
純粋な反時計回り回転では $\theta \approx \pi/2$ に集中するはずで、その近傍にあることを確認。

**0.3 という許容幅の理由**:  
実測値 ≈ 1.775 に対して π/2 ≈ 1.571 で差は約 0.2。この系統的バイアスは：

1. **有限差分バイアス**：`omega=0.2` の場合、1 step の時間遅れで +0.1 rad のオフセット
2. **正規化バイアス**：PC 間の分散が揃わない場合に jPC 平面が楕円状になり角度がシフト

どちらもノイズではなく系統的なバイアスのため、許容幅 0.3 rad（≈ 17°）でカバーしている。

---

## Test 8 — ランダムデータの $R^2_\text{ratio}$ は回転データより低い

**確認内容**: `R2_ratio(random) < R2_ratio(rotation)`

**何を確認しているか**:  
jPCA が「構造のあるデータ」と「ランダムノイズ」を区別できることを確認する比較テスト。  

- 回転データ：$R^2_\text{ratio} > 0.85$（Test 6 より）
- ランダムデータ：$R^2_\text{ratio} \approx 0.5$（ランダム行列の対称成分と歪対称成分が平均的に等しい説明力を持つ）

理論的には両者の期待値差は明確だが、乱数シードに依存するため等号（<）ではなく不等号でテスト。

---

## Test 9 — 非リスト入力でエラー

**確認内容**: `expect_error(jpca_fit(matrix(...)))`

**何を確認しているか**:  
`jpca_fit` が `list` のみを受け付け、行列を直接渡した場合に適切なエラーを出すことを
確認する入力バリデーションテスト。  
R では `list` と `matrix` は型が異なり、関数内で `is.list(X_list)` チェックをしている。

---

## Test 10 — `jpca_transform` の出力サイズ

**確認内容**:
- `proj$proj` の shape: `[2, C*T]`
- `proj$proj_list` の長さ: `C`
- 各 `proj_list[[c]]` の shape: `[2, T]`

**何を確認しているか**:  
射影の出力形状を確認する。`W` は `[2 × n_pcs]` なので、`W %*% X_red [n_pcs × C*T]` = `[2 × C*T]`。
これを条件ごとに分割すれば各条件で `[2 × T]`。

形状確認は「計算が意図した軸で行われているか」の基本チェックで、転置ミスなどの低レベルバグを
防ぐためのものです。

---

## まとめ

| # | テスト | 確認内容 | 数学的根拠 |
|---|---|---|---|
| 1 | M_skew 歪対称 | $M + M^\top = 0$ | 定義から自明 |
| 2 | 固有値が純虚数 | $\text{Re}(\lambda) \approx 0$ | Skew-symmetric の性質 |
| 3 | jPC1 ⊥ jPC2 | 内積 ≈ 0 | 複素固有ベクトルの Re/Im 直交性 |
| 4 | 単位ベクトル | ノルム = 1 | 明示的正規化 |
| 5 | $R^2_\text{skew} \leq R^2_\text{unrestr}$ | 制約付き ≤ 制約なし | 最小化の包含関係 |
| 6 | $R^2_\text{ratio} > 0.85$ | 回転性の強さ | ノイズが小さければほぼ 1 |
| 7 | peak ≈ π/2（±0.3） | θ 分布のピーク | 反時計回り回転の幾何 |
| 8 | random < rotation | 比較テスト | ランダム期待値 ≈ 0.5 < 0.85 |
| 9 | エラー（非リスト入力） | 入力バリデーション | — |
| 10 | 出力 shape 確認 | [2,C*T], [2,T] | 行列積の次元計算 |
