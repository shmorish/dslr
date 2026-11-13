# dslr

Harry Potter and the Data Scientist（42 dslr）。Python + uv。

```sh
uv sync
uv run python analysis/describe.py datasets/dataset_train.csv
uv run pytest                # テスト
```

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
| `visualization/pair_plot.py` | ロジスティック回帰に使う特徴量は？ | 上の3つ（均質2科目と重複1科目）を除いた10科目 |

## ロジスティック回帰（one-vs-all）

```sh
uv run python logreg/logreg_train.py datasets/dataset_train.csv            # -> weights.json
uv run python logreg/logreg_predict.py datasets/dataset_test.csv weights.json  # -> houses.csv
```

- 前処理: 欠損値は学習データの平均で補完し、標準化する（平均・標準偏差は weights.json に保存）
- 学習: 寮ごとに sigmoid + log loss を勾配降下で最小化（lr 0.5, 1000 epoch）
- 予測: 4つのモデルのスコアが最大の寮
- 精度: 学習データ 98.2%（残りは寮の塊から外れたラベルノイズ）、test 99.0%

## ボーナス

- `analysis/describe.py`: Var / Range / IQR / Skew / Missing（欠損数）を追加
- `logreg/logreg_train.py dataset_train.csv [batch|minibatch|sgd]`: バッチ GD（既定）、mini-batch GD（32件）、SGD（1件）を選択可。どれも test 99.0%
