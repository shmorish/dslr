import json
import sys

from dslr.io import HOUSES, load_houses
from dslr.model import ALGOS, predict, train


def main():
    usage = f"usage: logreg_train.py dataset_train.csv [{'|'.join(ALGOS)}]"
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in ALGOS):
        sys.exit(usage)
    algo = sys.argv[2] if len(sys.argv) == 3 else "batch"
    cols, houses = load_houses(sys.argv[1])
    keep = [i for i, h in enumerate(houses) if h in HOUSES]
    cols = {f: [v[i] for i in keep] for f, v in cols.items()}
    houses = [houses[i] for i in keep]
    model = train(cols, houses, algo)
    with open("weights.json", "w") as f:
        json.dump(model, f, indent=2)
    acc = sum(p == h for p, h in zip(predict(model, cols), houses)) / len(houses)
    print(f"[{algo}] weights saved to weights.json (training accuracy {acc:.4f})")


if __name__ == "__main__":
    main()
