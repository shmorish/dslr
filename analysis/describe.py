import sys

from dslr.io import load_csv, numeric_columns
from dslr.stats import STATS, describe


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: describe.py dataset.csv")
    header, rows = load_csv(sys.argv[1])
    table = describe(numeric_columns(header, rows, pandas_like=True))
    if not table:
        sys.exit("error: no numerical features")
    stats = [s for s, _ in STATS] + ["Missing"]
    cells = {n: [f"{table[n][s]:.6f}" for s in stats] for n in table}
    widths = {n: max(len(n), *(len(c) for c in cells[n])) + 2 for n in table}
    print(" " * 8 + "".join(n.rjust(w) for n, w in widths.items()))
    for i, stat in enumerate(stats):
        print(stat.ljust(8) + "".join(cells[n][i].rjust(w) for n, w in widths.items()))


if __name__ == "__main__":
    main()
