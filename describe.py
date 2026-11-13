import sys

from utils import count, load_csv, maximum, mean, minimum, numeric_columns, percentile, std

STATS = [
    ("Count", count),
    ("Mean", mean),
    ("Std", std),
    ("Min", minimum),
    ("25%", lambda xs: percentile(xs, 25)),
    ("50%", lambda xs: percentile(xs, 50)),
    ("75%", lambda xs: percentile(xs, 75)),
    ("Max", maximum),
]


def describe(cols):
    """{feature: {stat: value}}; NaN for stats undefined on too few values."""
    table = {}
    for name, values in cols.items():
        xs = [v for v in values if v is not None]
        table[name] = {}
        for stat, fn in STATS:
            try:
                table[name][stat] = float(fn(xs))
            except (ZeroDivisionError, IndexError):
                table[name][stat] = float("nan")
    return table


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: describe.py dataset.csv")
    header, rows = load_csv(sys.argv[1])
    table = describe(numeric_columns(header, rows))
    if not table:
        sys.exit("error: no numerical features")
    widths = {n: max(14, len(n)) + 2 for n in table}
    print(" " * 8 + "".join(n.rjust(w) for n, w in widths.items()))
    for stat, _ in STATS:
        print(stat.ljust(8) + "".join(f"{table[n][stat]:{w}.6f}" for n, w in widths.items()))


if __name__ == "__main__":
    main()
