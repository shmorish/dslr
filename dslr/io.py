import csv
import math
import sys

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


def load_csv(path):
    """Return (header, rows) or exit with a readable error."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        sys.exit(f"error: cannot read {path}: {e}")
    if not rows:
        sys.exit(f"error: {path} is empty")
    return rows[0], rows[1:]


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def is_missing(cell):
    """Empty, nan and inf cells carry no usable value."""
    if cell == "":
        return True
    f = to_float(cell)
    return f is not None and not math.isfinite(f)


def numeric_columns(header, rows, pandas_like=False):
    """{name: [float|None]} for every column whose non-missing values all parse as numbers.

    Features only by default. pandas_like also keeps Index and all-empty columns,
    which pandas.read_csv(...).describe() reports too.
    """
    cols = {}
    for j, name in enumerate(header):
        if name == "Index" and not pandas_like:
            continue
        cells = [r[j] if j < len(r) else "" for r in rows]
        present = [c for c in cells if not is_missing(c)]
        if (present or pandas_like) and all(to_float(c) is not None for c in present):
            cols[name] = [None if is_missing(c) else to_float(c) for c in cells]
    return cols


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


def require_columns(cols, names, path):
    missing = [n for n in names if n not in cols]
    if missing:
        sys.exit(f"error: {path} is missing numeric columns {missing}")
