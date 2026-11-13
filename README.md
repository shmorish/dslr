# dslr

Harry Potter and the Data Scientist（42 dslr）。Python + uv。

```sh
uv sync
uv run python describe.py datasets/dataset_train.csv
uv run python test_dslr.py   # self-check
```

## 可視化（第2引数に PNG パスを渡すと保存）

| スクリプト | 問い | 答え |
|---|---|---|
| `histogram.py` | 4寮で成績分布が均質な科目は？ | Arithmancy / Care of Magical Creatures |
| `scatter_plot.py` | 似ている2特徴量は？ | Astronomy と Defense Against the Dark Arts（相関 -1） |
| `pair_plot.py` | ロジスティック回帰に使う特徴量は？ | 上の3つ（均質2科目と重複1科目）を除いた10科目 |
