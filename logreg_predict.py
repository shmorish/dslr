import csv
import json
import sys

from logreg_train import predict
from utils import load_csv, numeric_columns


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: logreg_predict.py dataset_test.csv weights.json")
    header, rows = load_csv(sys.argv[1])
    try:
        with open(sys.argv[2]) as f:
            model = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"error: cannot load weights: {e}")
    cols = numeric_columns(header, rows)
    missing = [f for f in model["features"] if f not in cols]
    if missing:
        sys.exit(f"error: missing feature columns {missing}")
    index = header.index("Index") if "Index" in header else None
    with open("houses.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["Index", "Hogwarts House"])
        for i, (row, house) in enumerate(zip(rows, predict(model, cols))):
            w.writerow([row[index] if index is not None else i, house])
    print("predictions saved to houses.csv")


if __name__ == "__main__":
    main()
