# dPCA（demixed PCA）ノート
### 数学的基盤の詳細説明 — 実装情報を含まない

> このノートはdPCAの数学的理解のためのものです。実装・コードは含みません。
> **jPCA_notes.md を先に読んでください。** dPCAはjPCAで使う「task-specific subspace」を定義する手法です。
> 疑問が生じた箇所には **[Q]** を、理解確認には **[✓]** を付けて更新してください。

---

## 1. なぜ PCA では不十分か

標準的なPCAは「分散を最大化する軸」を見つける。
問題は、その軸が複数のタスク変数を混ぜてしまうことです：

```
ニューロン活動 x(t) は rule と stim と resp の混合物
→ PCA の第1主成分は「rule × stim × resp × time の合成」になる
→ 「rule subspace」「stim subspace」が分離できない

例）PC1 が高いとき、それは
  (a) ruleが変わったから？
  (b) stimが変わったから？
  (c) 両方？
  → PCAには区別する手段がない
```

dPCAはこの問題を解決します：
各タスクパラメータ（rule, stim, time, ...）に起因する分散を**分離**し、それぞれを最もよく説明する軸を求めます。

---

## 2. dPCAの核心アイデア

### 一言で言うと

```
まず X を「rule が原因の部分」「stim が原因の部分」...に分解する（ANOVA）
それぞれの部分に対して PCA をかける
→ 各パラメータに「demixされた」コンポーネントが得られる
```

### 数式で

```
X = X_t + X_r + X_s + X_tr + X_ts + X_rs + X_trs   （ANOVA分解）
        ↓           ↓           ↓
  timeだけで    ruleだけで   stimだけで
  変化する部分  変化する部分  変化する部分

各 X_φ に SVD をかける：
  SVD(X_φ) → D_φ（demixed decoder）

射影：Z_φ = D_φᵀ X  ← 「ruleに関する表現」「stimに関する表現」が分離される
```

---

## 3. 基礎知識：2要因ANOVAとの対応

dPCAのANOVA分解は**2要因ANOVAの効果分解**と全く同じ考え方です。

### 2要因ANOVAの場合（factor A, B）

```
観測値 = 全体平均 + Aの主効果 + Bの主効果 + A×B交互作用 + 誤差

Y_ij = μ + α_i + β_j + (αβ)_ij + ε_ij
```

### dPCAの場合（time t, rule r, stim s）

```
X[n, t, r, s] = μ[n]      （全体平均）
              + X_t[n, t]  （timeの主効果）
              + X_r[n, r]  （ruleの主効果）
              + X_s[n, s]  （stimの主効果）
              + X_tr[n,t,r]（time×rule交互作用）
              + X_ts[n,t,s]（time×stim交互作用）
              + X_rs[n,r,s]（rule×stim交互作用）
              + X_trs[n,t,r,s]（3要因交互作用）

N = ニューロン数（またはEEGチャンネル数）は「繰り返し」に相当
各 X_φ はその主効果または交互作用だけを含む「純粋な成分」
```

---

## 4. ANOVA分解（マージナライゼーション）の計算方法

これがdPCAで最も重要かつ最も難しい部分です。

### Step-by-step（labels = "ts"：time と stim の2要因の場合）

データ：`X[N, T, S]`  （N=ニューロン、T=time、S=stim）

```
Step A：全体平均を計算
  μ[n] = mean over all T, S

Step B：各パラメータの条件平均を計算
  mean_S(X)[n, t]  ← stimを平均したもの（t だけが残る）
  mean_T(X)[n, s]  ← timeを平均したもの（s だけが残る）

Step C：純粋な主効果を引き算で取り出す
  X_t[n, t]  = mean_S(X)[n, t] - μ[n]
                 ↑「timeで変化するが stimで変化しない」成分

  X_s[n, s]  = mean_T(X)[n, s] - μ[n]
                 ↑「stimで変化するが timeで変化しない」成分

Step D：交互作用を取り出す（下位の効果をすべて引く）
  X_ts[n,t,s] = X[n,t,s] - X_t[n,t] - X_s[n,s] - μ[n]
                 ↑「t と s が同時に変化するときだけ起きる」成分
```

### 確認：和が元のデータになる

```
X_t + X_s + X_ts = X - μ  （Frobenius誤差 < 1e-10 で成立するはず）

各成分は「直交」（Frobenius内積がゼロ）：
  <X_t, X_s>_F = 0
  <X_t, X_ts>_F = 0
  <X_s, X_ts>_F = 0
```

### 3要因の場合（labels = "trs"：time, rule, stim）

```
サブセットの列挙（7つ）：t, r, s, tr, ts, rs, trs

計算順序：小さいサブセットから順番に
  X_t   = mean_{r,s}(X) - μ
  X_r   = mean_{t,s}(X) - μ
  X_s   = mean_{t,r}(X) - μ
  X_tr  = mean_s(X) - X_t - X_r - μ
  X_ts  = mean_r(X) - X_t - X_s - μ
  X_rs  = mean_t(X) - X_r - X_s - μ
  X_trs = X - X_t - X_r - X_s - X_tr - X_ts - X_rs - μ
```

**重要：** 各 X_φ を計算するとき、それより小さいすべてのサブセットの効果を引く（包除原理）。
これを省くと成分が混ざってしまう。

### なぜこの手順で直交するか

```
X_t は「t方向にのみ変化する」→ s方向の変化成分がゼロ
X_s は「s方向にのみ変化する」→ t方向の変化成分がゼロ
→ X_t と X_s を列方向に内積を取ると、直交する対称性から和がゼロになる

これは ANOVAの「効果の非交絡性」と同じ原理
```

---

## 5. Decoder の計算（SVD）

X_φ を [N × K] に unfold（Kは条件数の積）してから SVD をかけます：

```
SVD( X_φ[N × K] ) = U Σ Vᵀ

D_φ = U[:, 1:k_φ]   ← top k_φ 左特異ベクトル

意味：X_φ の分散を最大化する方向の軸
     = 「factor φ に起因する分散を最もよく捉える」軸
     = 「demixed PCA components」
```

### 標準PCAとの違い

```
標準PCA：  X     の分散最大化 → components が全因子を混ぜる
dPCA：    X_φ   の分散最大化 → components が factor φ だけに関する分散を捉える
```

---

## 6. Encoder の計算（Biorthogonal regression）

DecoderD_φはデータを低次元に射影する。
Encoderは低次元から元の空間に「戻す」ための行列。

### 射影

```
Z_φ = D_φᵀ X_2d   [k_φ × K]  ← factor φ の latent representation
```

ここで `X_2d` は元の全データ（X_φ ではなく X そのもの）を [N × K] にunfoldしたもの。

### Encoderの導出

```
X_φ ≈ F_φ Z_φ

最小二乗解：
  F_φ = X_φ Zᵀ_φ (Z_φ Zᵀ_φ)⁻¹   [N × k_φ]
```

### Biorthogonality（双直交性）

```
D_φᵀ F_φ ≈ I_{k_φ}

意味：「factor φ の latent Z_φ を D_φ で射影した成分が、
       F_φ で元の空間に戻したときに正確に対応する」

なぜ重要か？
  標準PCAでは D_φ = F_φ（decoder と encoder が同じ）
  しかし D_φ は X_φ を最大化するように決まっており、
  X（全データ）を射影すると他の因子の分散も混じる
  → F_φ ≠ D_φ が必要
  → biorthogonality が、各 latent 次元が「1つの因子だけに対応する」ことを保証
```

---

## 7. Regularization（正則化）

N > K のとき（ニューロン数 > 条件数×時間数）、Z_φ Z_φᵀ は特異行列になり逆行列が存在しない。

```
F_φ = X_φ Zᵀ_φ (Z_φ Zᵀ_φ + αI)⁻¹

α = regularizer（正則化パラメータ）

典型的なEEGデータ（N = 150, K = T×C = 100×12 = 1200）では
  N < K なので α = NULL（正則化不要）で十分
  N > K になる場合は α を cross-validation で選ぶ
```

---

## 8. 有意性検定（Significance）

各 component が有意かどうかを shuffle test で確認：

```
1. X_trial（単試行データ）を train/test に分割
2. train で D_φ を fit
3. test の Z_φ = D_φᵀ X_test に対して 1-NN 分類器を適用
4. 分類スコアを計算
5. condition label をシャッフルして null distribution を作る
6. 真のスコア > null の最大値 → significant（p < 0.05）
```

---

## 9. dPCA → jPCA への接続

```
dPCA で得られた Z_r（rule subspace）、Z_s（stim subspace）...
に対して jPCA をかける：

jpca_fit( list of condition matrices from Z_r )

問い：「ruleに関する表現」は時間的に回転するか？
      ruleとstimを混ぜた表現より、rule subspaceだけの方が
      rotation が強く見えるか？
```

---

## 10. アルゴリズムのまとめ

```
Input:
  X_avg   [N × T × n_r × n_s]  trial-averaged EEG
  labels  "trs"
  n_components: int または named list (t=3, r=3, s=3, rs=2)

────────────────────────────────────────────────────

Step 1: ANOVA 分解（dpca_marginalize）

  全体平均 μ を引く
  各サブセット φ について X_φ を計算（Section 4参照）
  X_φ を [N × K] に unfold

  検証：sum(X_φ) ≈ X_2d - μ  （Frobenius < 1e-10）

────────────────────────────────────────────────────

Step 2: Decoder 計算（SVD）

  for each φ:
    g(U, s, V) %=% svd(X_φ)
    D_φ = U[, 1:k_φ]
    var_exp_φ = s^2 / sum(s^2)

────────────────────────────────────────────────────

Step 3: Encoder 計算

  for each φ:
    Z_φ = t(D_φ) %*% X_2d           ← 全データを射影
    F_φ = X_φ %*% t(Z_φ) %*% solve(Z_φ %*% t(Z_φ) + α*I)

  検証：t(D_φ) %*% F_φ ≈ I_{k_φ}   （biorthogonality）

────────────────────────────────────────────────────

Step 4: 射影（Transform）

  Z_φ = t(D_φ) %*% X_2d
  reshape to [k_φ × T × n_r × n_s]（元の条件次元に戻す）

────────────────────────────────────────────────────

Output:
  D_φ      : decoders（各因子の demixed components）
  F_φ      : encoders（biorthogonal）
  Z_φ      : projected data（各因子の latent representation）
  var_exp  : explained variance ratio per marginalization
```

---

## 11. Comprehension Check

実装前にすべて答えられること：

1. **なぜ X_ts の計算で X_t と X_s を引くのか？**
   ヒント：X_ts = mean_r(X) とだけ定義すると、X_ts には time の主効果も含まれてしまう。

2. **X_t と X_s が Frobenius 直交することを手で確認せよ（2×3×2 の toy data で）**
   ヒント：`<X_t, X_s>_F = trace(X_tᵀ X_s)` を計算する。

3. **なぜ D_φ の計算で X_φ を使い、Z_φ の計算で X_2d（全データ）を使うのか？**
   ヒント：D_φ は「factor φ の分散を最大化する軸」であって、「factor φ だけを含む空間への射影」ではない。

4. **biorthogonality D_φᵀ F_φ ≈ I はなぜ必要か？**
   ヒント：D_φ = F_φ（PCA的な使い方）にした場合、Z_φ = D_φᵀ X に含まれる分散は何に由来するか？

5. **N < K のとき regularizer が不要な理由は？**
   ヒント：Z_φ Z_φᵀ が何×何の行列か、それが可逆かを考える。

6. **dPCA → jPCA のパイプラインで、なぜ Z_r（rule subspace）に jPCA をかけることが意味を持つか？**
   ヒント：raw EEGに直接 jPCA をかけたとき、rotation は何の因子によって引き起こされているかわからない。

---

## 12. 参考文献

- **Kobak, D. et al. (2016).** Demixed principal component analysis of neural population data. *eLife* 5:e10989.
  - dPCAのオリジナル論文。Methods section に全アルゴリズムの詳細。
  - Fig.2 が surrogate data でのデモ、Fig.3-5 が実データへの適用。
- **Brendel, W. et al. (2011).** Demixed principal component analysis. *NIPS 2011.*
  - 先行論文（前身）。基本的なアイデアが先に提示されている。
- **machenslab/dPCA** (Python/MATLAB). https://github.com/machenslab/dPCA
  - Reference implementation。数値検証に使用。
