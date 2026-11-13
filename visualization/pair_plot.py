import sys

import matplotlib.pyplot as plt

from dslr.io import HOUSES, load_houses
from dslr.plot import COLORS, by_house, show_or_save

# Features kept for logistic regression: drop Arithmancy and Care of Magical Creatures
# (houses indistinguishable) and Defense Against the Dark Arts (duplicate of Astronomy).


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: pair_plot.py dataset_train.csv [out.png]")
    cols, houses = load_houses(sys.argv[1])
    names = list(cols)
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
    for i, yname in enumerate(names):
        for j, xname in enumerate(names):
            ax = axes[i][j]
            for h in HOUSES:
                if i == j:
                    ax.hist(by_house(cols[xname], houses, h), bins=20, alpha=0.5, color=COLORS[h])
                else:
                    pts = [(x, y) for x, y, hh in zip(cols[xname], cols[yname], houses)
                           if hh == h and x is not None and y is not None]
                    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=1, alpha=0.5, color=COLORS[h])
            ax.set_xticks([])
            ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(xname[:12], fontsize=7)
            if j == 0:
                ax.set_ylabel(yname[:12], fontsize=7)
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COLORS[h], label=h) for h in HOUSES],
               loc="upper right")
    fig.tight_layout()
    show_or_save(plt, sys.argv)


if __name__ == "__main__":
    main()
