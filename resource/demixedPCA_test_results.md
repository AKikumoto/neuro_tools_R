# demixed PCA テスト結果の説明

**テストスクリプト**: `test/test_demixedPCA.R`（15 テスト、全 PASS）  
**実行方法**: テストランナーで `demixedPCA_lib.R` を先に source してから実行  
**関連ノート**: [demixedPCA_notes.md](demixedPCA_notes.md)

> このドキュメントはテストスクリプトの結果を説明するものです。各テストが何を確認しているか、
> なぜその条件が成り立つかを記述します。実装コードは含みません。

---

## サーロゲートデータ

### `make_demo_data`（Kobak 2016 の dPCA デモに準拠）

- サイズ：`X[N=100, S=6, T=250]`（N: チャンネル、S: stimulus 数、T: timepoints）
- 2 つの潜在因子：
  - $z_t = t/T$（linear ramp in time）
  - $z_s = s/S$（linear spacing over stimuli）
- チャンネルごとに ランダム係数 $a_t, a_s \sim \mathcal{N}(0,1)$：
  $X[n,s,t] = a_t z_t[t] + a_s z_s[s] + \varepsilon$
- ノイズ：$\varepsilon \sim \mathcal{N}(0, 0.2^2)$

このデータは stimulus 主効果と time 主効果が加法的に混在し、交互作用はない（理想的な dPCA テストデータ）。

---

## Test 1 — `get_marginalizations` の構造

**確認内容**:
- `length(m) == 3`（$2^2 - 1 = 3$ 個のサブセット）
- `names(m) == c("s", "t", "st")`（辞書順）
- `m[["s"]] == 0L`、`m[["t"]] == 1L`、`m[["st"]] == c(0L, 1L)`

**なぜ成り立つか**:  
`dpca_get_marginalizations("st")` は 2 文字のラベル文字列を受け取り、
7 つ（$2^2 - 1 = 3$）の非空サブセットを列挙する。各サブセットの値は
`0-indexed` の軸番号（R の 0 ベース）を示し、ラベル名と一致する。

---

## Test 2 — マージナリゼーションの合計が中心化 $\tilde{X}$ に等しい

**確認内容**: `max(|sum(X_phi) - Xcentered|) < 1e-10`

**なぜ成り立つか**:  
包除原理による分解が正しく実装されていれば：
$$X^{(s)} + X^{(t)} + X^{(st)} = \tilde{X}$$

各 $X^{(\phi)}$ の計算で下位サブセットの効果を引いているため、全成分の総和が
中心化データに一致する。数値誤差は浮動小数点演算の丸め誤差（$10^{-14}$ オーダー）のみ。

---

## Test 3 — マージナリゼーションの相互直交性

**確認内容**: 全ペア $(s, t), (s, st), (t, st)$ で Frobenius 内積 $< 10^{-6}$

**なぜ成り立つか**:  
$X^{(s)}$ は「s 方向にのみ変化し t 方向の変化がゼロ」、$X^{(t)}$ は「t 方向にのみ変化し s 方向の変化がゼロ」。
これら異なる方向に特化した行列の内積は、対称性から相殺してゼロになる（ANOVA の非交絡性）。

理論的には厳密にゼロだが、浮動小数点誤差で $10^{-12}$ 程度の残余があるため、
閾値を $10^{-6}$ に設定している。

---

## Test 4 — `dpca_fit` の P と D の shape

**確認内容**: 各マージナリゼーション $\phi$ で `dim(P[[φ]]) == c(N, k)` かつ `dim(D[[φ]]) == c(N, k)`

**なぜ成り立つか**:  
`dpca_fit` では各 $\phi$ について SVD を計算し、上位 $k$ 列を返す：
- $F_\phi = U[:,1:k]$（$[N \times k]$）
- $D_\phi = C_\phi^\top U[:,1:k]$（$[N \times k]$）

shape の確認はバグ（転置忘れ・axis の混同など）の早期検出に重要。

---

## Test 5 — `dpca_transform` の出力 shape

**確認内容**: 各 $\phi$ で `dim(Z[[φ]]) == c(k, S, T_len)`

**なぜ成り立つか**:  
射影 $Z_\phi = D_\phi^\top \tilde{X}_\text{flat}$ は $[k \times (S \cdot T)]$ を返し、
これを元の条件次元 $[k \times S \times T]$ に reshape する。
R の配列は列優先（column-major）なので、reshape の順序が正しいことへの確認でもある。

---

## Test 6 — 説明分散の非負性と降順ソート

**確認内容**: `ev[[φ]] >= -1e-10`（非負）かつ `diff(ev[[φ]]) <= 1e-10`（降順）

**なぜ成り立つか**:  
$R^2_{\phi,j} = \|D_{\phi,j}^\top \tilde{X}\|^2 / \|\tilde{X}\|_F^2$。
分子は 2-ノルムの 2 乗なので非負。
SVD の特異値は降順であり、説明分散も対応して降順になる（上位成分ほど大きい分散を説明）。

---

## Test 7 — 第 1 stimulus コンポーネントは stimulus 分散を捉える

**確認内容**: `var(rowMeans(z1)) > var(colMeans(z1)) * 0.5`

**何を確認しているか**:  
`Z[["s"]][1,,]`（$[S \times T]$ 行列）の：
- `rowMeans(z1)` = 各 stimulus の時間平均 → stimulus の違いによる分散
- `colMeans(z1)` = 各 timepoint の stimulus 平均 → time の違いによる分散

stimulus コンポーネントなのだから、stimulus 方向の分散が time 方向の分散と
少なくとも同程度（係数 0.5 以上）であることを確認する。

**0.5 の根拠**：デマックスが完璧な理想データではなく、ノイズ（$\sigma=0.2$）の影響で
多少 time 分散が混入する可能性を許容している。

---

## Test 8 — 第 1 time コンポーネントは time 分散を捉える

**確認内容**: `var(colMeans(z1)) > var(rowMeans(z1)) * 0.5`

**何を確認しているか**:  
Test 7 の逆：`Z[["t"]][1,,]` では time 方向の分散が支配的であることを確認。
サーロゲートデータの time 因子（linear ramp $z_t$）が正しく捉えられているかのチェック。

---

## Test 9 — stimulus + time の説明分散が交互作用より大きい

**確認内容**: `sum(ev[["s"]]) + sum(ev[["t"]]) > sum(ev[["st"]])`

**なぜ成り立つか**:  
サーロゲートデータは $X = a_t z_t + a_s z_s + \varepsilon$ で生成されており、
stimulus と time が**加法的**な構造を持つ。交互作用成分 $X^{(st)}$ は
残差に相当し、小さいはず。

この比較は「dPCA が構造を正しく分解しているか」の間接的な確認。

---

## Test 10 — `dpca_inverse_transform` の出力 shape

**確認内容**: `dim(Xrec) == c(N, S, T_len)`

**なぜ成り立つか**:  
$\hat{X}^{(s)} = F_s Z_s = F_s D_s^\top \tilde{X}_\text{flat}$。  
$F_s$ は $[N \times k]$、$Z_s$ は $[k \times S \cdot T]$ なので、積は $[N \times S \cdot T]$。
これを $[N \times S \times T]$ に reshape して返す。  
（過去のバグ：`nrow` を `ncol(Pk)` でなく `nrow(Pk)=N` にすることで shape ミスが起きた。このテストで検出された。）

---

## Test 11 — 全マージナリゼーションの再構成が $\tilde{X}$ に近い

**確認内容**: `rel_err = ||Xrec_sum - Xc||_F^2 / ||Xc||_F^2 < 0.5`

**何を確認しているか**:  
$\sum_\phi F_\phi Z_\phi \approx \tilde{X}$ が成り立つかどうか（`n_components = 6` のとき）。
厳密には $k < \text{rank}(\tilde{X})$ なので完全な一致ではなく、
相対誤差 50% 未満という緩い条件でテストする。

**0.5 という閾値の根拠**：コンポーネント数 6 で各マージナリゼーション 3 成分は全分散を
カバーできない。完全な再構成を要求するのではなく、「大きく外れていない」ことを確認する。

---

## Test 12 — 3 ラベルケースで 7 個のマージナリゼーション

**確認内容**: `length(m) == 7L` かつ `names(m) == c("s","t","c","st","sc","tc","stc")`

**なぜ成り立つか**:  
$K = 3$ のとき $2^3 - 1 = 7$ 個の非空サブセットが生成される。
名前の順序はサブセット長ソート → 辞書順：
1 文字（3 個）→ 2 文字（3 個）→ 3 文字（1 個）。

このテストは `dpca_get_marginalizations` の enumerate が正しく機能するかを確認する。

---

## Test 13 — 正則化があっても P/D の shape は変わらない

**確認内容**: `regularizer = 0.01` を指定しても `dim(P[[φ]]) == c(N, k)` が成立

**なぜ成り立つか**:  
正則化は $\tilde{X}_\text{flat}$ に $\lambda I_N$ 列を append するだけで、
SVD の実行後に上位 $k$ 列を返す手順は変わらない。shape は `regularizer` に依らない。

---

## Test 14 — `n_components` を named list で各マージナリゼーションに個別設定

**確認内容**: `nc = list(s=2, t=4, st=1)` のとき `ncol(P[["s"]])==2`, `ncol(P[["t"]])==4`, `ncol(P[["st"]])==1`

**何を確認しているか**:  
`dpca_fit` が `n_components` に named list を受け付け、マージナリゼーションごとに
異なるコンポーネント数で SVD を切り捨てる機能が正しく動くかを確認する。

これは実用上重要：「stimulus は 2 成分、time は 4 成分でカバー」のように
データの複雑さに応じて調整できる。

---

## Test 15 — stimulus の説明分散 > 交互作用の説明分散

**確認内容**: `ev[["s"]][1] > ev[["st"]][1]`（第 1 成分同士の比較）

**なぜ成り立つか**:  
サーロゲートデータには交互作用がなく（$X^{(st)} \approx 0$）、stimulus 主効果が
明確に存在する。したがって stimulus マージナリゼーションの第 1 成分の説明分散は、
交互作用マージナリゼーションの第 1 成分より大きくなる。

これは dPCA が「構造のある因子を小さなコンポーネント数で効率よく説明できる」という
モデルの核心的な性質を確認するテスト。

---

## まとめ

| # | テスト | 確認内容 | 数学的根拠 |
|---|---|---|---|
| 1 | `get_marginalizations` 構造 | $2^K - 1$ 個のサブセット | 組み合わせ論 |
| 2 | 合計 = $\tilde{X}$ | 分解の完全性 | 包除原理 |
| 3 | Frobenius 直交 | 各成分の独立性 | ANOVA の非交絡性 |
| 4 | P, D の shape `[N, k]` | SVD 出力の形状 | 行列積の次元 |
| 5 | Z の shape `[k, S, T]` | reshape の正確さ | 列優先配列の変換 |
| 6 | EV 非負・降順 | 分散説明量の単調性 | SVD 特異値の性質 |
| 7 | Z[s] が stimulus 分散を捉える | demix の有効性（s 側） | 主効果分離 |
| 8 | Z[t] が time 分散を捉える | demix の有効性（t 側） | 主効果分離 |
| 9 | s + t EV > st EV | 加法的構造の検出 | 交互作用の小ささ |
| 10 | inverse_transform shape | 再構成の次元 | $F D^\top$ の積 |
| 11 | 再構成誤差 < 50% | 近似品質 | 有限成分での近似 |
| 12 | 3 ラベルで 7 成分 | $2^3 - 1 = 7$ | 組み合わせ論 |
| 13 | 正則化で shape 不変 | 実装の独立性 | 列 append 後の SVD |
| 14 | named list の k 設定 | 柔軟なコンポーネント数 | — |
| 15 | s EV > st EV | 主効果 > 交互作用 | 加法的データの性質 |
