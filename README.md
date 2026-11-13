# dslr

Harry Potter and the Data Scientist（42 dslr）。Python + uv。

```sh
uv sync
uv run python analysis/describe.py datasets/dataset_train.csv
uv run pytest                # テスト
```

## 環境

- Python 3.13（`.python-version`）と uv 0.13.1 を、ローカル・CI・devcontainer でそろえている
- devcontainer: VS Code の「Reopen in Container」か `devcontainer up --workspace-folder .`。コンテナ内には画面がないので、可視化スクリプトには保存先の PNG パスを渡す
- CI: pytest と学習→予測の通し実行、`actions.lock` の検証。action を変えたら `gh actions-lock` で lockfile を更新する
- Dependabot が action の更新 PR を週1で作り、同じ PR で `actions.lock` も更新する

## 構成

```
datasets/        学習・テストデータ
dslr/            共有ライブラリ（uv sync で editable install）
  io.py          CSV 読み込み・数値列抽出・寮ラベル
  stats.py       自前の統計関数と describe の表
  plot.py        寮の色・表示/保存
  model.py       sigmoid・勾配降下・train/predict
analysis/        V.1 describe.py
visualization/   V.2 histogram.py / scatter_plot.py / pair_plot.py
logreg/          V.3 logreg_train.py / logreg_predict.py
tests/           pytest
```

## 可視化（第2引数に PNG パスを渡すと保存）

| スクリプト | 問い | 答え |
|---|---|---|
| `visualization/histogram.py` | 4寮で成績分布が均質な科目は？ | Arithmancy / Care of Magical Creatures |
| `visualization/scatter_plot.py` | 似ている2特徴量は？ | Astronomy と Defense Against the Dark Arts（相関 -1） |
| `visualization/pair_plot.py` | ロジスティック回帰に使う特徴量は？ | 上の3つ（均質2科目と重複1科目）を除いた10科目。図でも使う科目は黒、除外した科目はグレーのラベルで示す |

## ロジスティック回帰（one-vs-all）

```sh
uv run python logreg/logreg_train.py datasets/dataset_train.csv            # -> weights.json
uv run python logreg/logreg_predict.py datasets/dataset_test.csv weights.json  # -> houses.csv
```

- 前処理: 欠損値は学習データの平均で補完し、標準化する（平均・標準偏差は weights.json に保存）
- 学習: 寮ごとに sigmoid + log loss を勾配降下で最小化（lr 0.5, 1000 epoch）。損失の推移を寮ごとに表示し、weights.json の `loss` にも残す
  ```
  Gryffindor  loss 0.6931 -> 0.0491 -> 0.0467 -> 0.0459 -> 0.0454 -> 0.0451  (epochs 1000)
  ```
- 予測: 4つのモデルのスコアが最大の寮
- 精度: 学習データ 98.2%（残りは寮の塊から外れたラベルノイズ）、test 99.0%

## ボーナス

- `analysis/describe.py`: Var / Range / IQR / Skew / Missing（欠損数）を追加
- `logreg/logreg_train.py dataset_train.csv [batch|minibatch|sgd]`: バッチ GD（既定）、mini-batch GD（32件）、SGD（1件）を選択可。どれも test 99.0%

## 設計メモ（評価で聞かれそうなこと）

### describe
- `count` / `mean` / `std` / `min` / `max` / `percentile` / ソートはすべて `dslr/stats.py` に自前で実装し、組み込みの `sum` / `sorted` / `len` などの集計関数は使っていない
- Std は標本標準偏差（n−1 で割る）、percentile は線形補間で、どちらも pandas の describe と同じ定義
- 空欄・`nan`・`inf` は欠損として扱い、Count から除く（欠損数はボーナス項目の Missing に出す）
- 出力する列は `pandas.read_csv(...).describe()` と同じ。`Index` も、test の空欄だけの `Hogwarts House`（Count 0）も出す。値の桁が大きくても列がくっつかないよう、列幅は整形後の文字数から決める

### ロジスティック回帰
- 仮説 `h(x) = g(θᵀx)`、`g(z) = 1 / (1 + e^(−z))` で、出力は「その寮である確率」
- 損失 `J(θ) = −(1/m) Σ [y log h + (1−y) log(1−h)]`、勾配 `∂J/∂θⱼ = (1/m) Σ (h(xⁱ) − yⁱ) xⱼⁱ` を使い、`θ ← θ − α∇J` で更新する
- one-vs-all: 寮ごとに「その寮か、それ以外か」の2値分類器を4つ学習し、確率が最大の寮を答えにする
- 標準化する理由: 科目ごとにスケールが大きく違う（Arithmancy は数万、Potions は一桁）ため、そのままだと勾配がスケールの大きい特徴に引っ張られて収束しない
- 欠損値は学習データの平均で埋める。行を捨てると予測時に答えを出せない生徒が出るため
- 特徴量の選び方: ヒストグラムで寮ごとの差がない2科目（Arithmancy / Care of Magical Creatures）と、散布図で Astronomy と完全相関の Defense Against the Dark Arts を除いた
- Batch GD は全データで1回更新、mini-batch は32件ごと、SGD は1件ごとに更新する。後者ほど1回の更新は粗いが更新回数が多く、少ないエポックで収束する
