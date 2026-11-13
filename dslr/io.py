import csv
import sys

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


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
