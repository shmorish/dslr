import sys

import matplotlib.pyplot as plt

from utils import COLORS, HOUSES, load_houses, show_or_save

# Answer: Astronomy and Defense Against the Dark Arts (correlation -1: DADA = -Astronomy / 100).
X, Y = "Astronomy", "Defense Against the Dark Arts"


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: scatter_plot.py dataset_train.csv [out.png]")
    cols, houses = load_houses(sys.argv[1])
    for h in HOUSES:
        pts = [(x, y) for x, y, hh in zip(cols[X], cols[Y], houses)
               if hh == h and x is not None and y is not None]
        plt.scatter([p[0] for p in pts], [p[1] for p in pts], s=8, alpha=0.6, color=COLORS[h], label=h)
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.title(f"Similar features: {X} vs {Y}")
    plt.legend()
    show_or_save(plt, sys.argv)


if __name__ == "__main__":
    main()
