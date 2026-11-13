import csv
import sys

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
COLORS = {"Gryffindor": "red", "Hufflepuff": "gold", "Ravenclaw": "blue", "Slytherin": "green"}


def load_csv(path):
    """Return (header, rows) or exit with a readable error."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except OSError as e:
        sys.exit(f"error: {e}")
    if not rows:
        sys.exit(f"error: {path} is empty")
    return rows[0], rows[1:]


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def numeric_columns(header, rows):
    """{name: [float|None]} for every column whose non-empty values all parse as numbers."""
    cols = {}
    for j, name in enumerate(header):
        if name == "Index":
            continue
        values = [to_float(r[j]) if j < len(r) and r[j] != "" else None for r in rows]
        non_empty = sum(1 for r in rows if j < len(r) and r[j] != "")
        parsed = [v for v in values if v is not None]
        if parsed and len(parsed) == non_empty:
            cols[name] = values
    return cols


def count(xs):
    n = 0
    for _ in xs:
        n += 1
    return n


def mean(xs):
    total = 0.0
    for x in xs:
        total += x
    return total / count(xs)


def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (count(xs) - 1)


def std(xs):
    return var(xs) ** 0.5


def minimum(xs):
    lo = xs[0]
    for x in xs:
        if x < lo:
            lo = x
    return lo


def maximum(xs):
    hi = xs[0]
    for x in xs:
        if x > hi:
            hi = x
    return hi


def percentile(xs, p):
    """Linear interpolation between closest ranks (same as numpy/pandas default)."""
    s = sorted(xs)
    k = (count(s) - 1) * p / 100
    lo = int(k)
    hi = lo + 1 if lo + 1 < count(s) else lo
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_houses(path):
    """(numeric columns, house per row) from a labelled dataset."""
    header, rows = load_csv(path)
    if "Hogwarts House" not in header:
        sys.exit(f"error: {path} has no 'Hogwarts House' column")
    j = header.index("Hogwarts House")
    houses = [r[j] for r in rows]
    if not any(h in HOUSES for h in houses):
        sys.exit(f"error: {path} has no house labels")
    return numeric_columns(header, rows), houses


def by_house(values, houses, house):
    return [v for v, h in zip(values, houses) if h == house and v is not None]


def show_or_save(plt, argv):
    """Show the figure, or save it to argv[2] when given (useful headless)."""
    if len(argv) > 2:
        plt.savefig(argv[2])
    else:
        plt.show()
