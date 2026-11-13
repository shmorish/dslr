import sys

import matplotlib.pyplot as plt

from dslr.io import HOUSES, load_houses
from dslr.plot import COLORS, by_house, show_or_save

# Answer: Arithmancy and Care of Magical Creatures — the four houses overlap almost exactly.
HOMOGENEOUS = ["Arithmancy", "Care of Magical Creatures"]


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: histogram.py dataset_train.csv [out.png]")
    cols, houses = load_houses(sys.argv[1])
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, name in zip(axes.flat, cols):
        for h in HOUSES:
            ax.hist(by_house(cols[name], houses, h), bins=25, alpha=0.5, color=COLORS[h], label=h)
        ax.set_title(name, fontweight="bold" if name in HOMOGENEOUS else "normal",
                     color="black" if name in HOMOGENEOUS else "gray", fontsize=9)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Homogeneous score distribution across houses: " + " / ".join(HOMOGENEOUS))
    fig.tight_layout()
    show_or_save(plt, sys.argv)


if __name__ == "__main__":
    main()
