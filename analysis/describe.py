import sys

from dslr.io import load_csv, numeric_columns
from dslr.stats import STATS, describe


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: describe.py dataset.csv")
    header, rows = load_csv(sys.argv[1])
    table = describe(numeric_columns(header, rows))
    if not table:
        sys.exit("error: no numerical features")
    widths = {n: max(14, len(n)) + 2 for n in table}
    print(" " * 8 + "".join(n.rjust(w) for n, w in widths.items()))
    for stat in [s for s, _ in STATS] + ["Missing"]:
        print(stat.ljust(8) + "".join(f"{table[n][stat]:{w}.6f}" for n, w in widths.items()))


if __name__ == "__main__":
    main()
