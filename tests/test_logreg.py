import csv
import json

import numpy as np
import pytest

from conftest import TEST, TRAIN
from logreg_train import ALGOS, FEATURES, gradient_descent, predict, sigmoid, train
from utils import HOUSES, load_houses


@pytest.fixture(scope="module")
def data():
    return load_houses(TRAIN)


def split(cols, houses, seed=0):
    idx = np.random.default_rng(seed).permutation(len(houses))
    cut = len(idx) * 4 // 5
    pick = lambda ids: ({f: [v[i] for i in ids] for f, v in cols.items()}, [houses[i] for i in ids])
    return pick(idx[:cut]), pick(idx[cut:])


def log_loss(X, y, theta):
    h = sigmoid(X @ theta)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(np.array([-50, 50])) == pytest.approx([0, 1], abs=1e-12)


@pytest.mark.parametrize("algo", ALGOS)
def test_gradient_descent_reduces_loss_on_separable_data(algo):
    rng = np.random.default_rng(1)
    X = np.hstack([np.ones((200, 1)), rng.normal(size=(200, 2))])
    y = (X[:, 1] + X[:, 2] > 0).astype(float)
    theta = gradient_descent(X, y, algo)
    assert log_loss(X, y, theta) < log_loss(X, y, np.zeros(3)) / 2
    assert np.mean((X @ theta > 0) == y) > 0.95


def test_gradient_descent_is_deterministic():
    X = np.hstack([np.ones((50, 1)), np.arange(50).reshape(-1, 1) / 50])
    y = (np.arange(50) > 25).astype(float)
    assert np.array_equal(gradient_descent(X, y, "sgd"), gradient_descent(X, y, "sgd"))


@pytest.mark.parametrize("algo", ALGOS)
def test_holdout_accuracy_at_least_98_percent(data, algo):
    (tr_cols, tr_h), (te_cols, te_h) = split(*data)
    pred = predict(train(tr_cols, tr_h, algo), te_cols)
    assert np.mean([p == h for p, h in zip(pred, te_h)]) >= 0.98


def test_model_contents(data):
    model = train(*data)
    assert model["features"] == FEATURES
    assert set(model["thetas"]) == set(HOUSES)
    assert all(len(t) == len(FEATURES) + 1 for t in model["thetas"].values())
    json.dumps(model)  # must be serialisable


def test_predict_fills_missing_values(data):
    model = train(*data)
    cols = {f: [None] for f in FEATURES}
    assert predict(model, cols)[0] in HOUSES


def test_train_rejects_missing_feature(data):
    cols, houses = data
    with pytest.raises(SystemExit, match="Astronomy"):
        train({f: v for f, v in cols.items() if f != "Astronomy"}, houses)


def test_cli_train_then_predict(run, tmp_path):
    r = run("logreg_train.py", TRAIN)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "weights.json").read_text())["features"] == FEATURES

    r = run("logreg_predict.py", TEST, "weights.json")
    assert r.returncode == 0, r.stderr
    with open(tmp_path / "houses.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["Index", "Hogwarts House"]
    assert [r[0] for r in rows[1:]] == [str(i) for i in range(400)]
    assert {r[1] for r in rows[1:]} <= set(HOUSES)
    assert (tmp_path / "houses.csv").read_bytes().count(b"\r") == 0


@pytest.mark.parametrize("algo", ALGOS)
def test_cli_train_algos(run, algo):
    r = run("logreg_train.py", TRAIN, algo)
    assert r.returncode == 0 and f"[{algo}]" in r.stdout


@pytest.mark.parametrize("script, args", [
    ("logreg_train.py", ()),
    ("logreg_train.py", (TRAIN, "adam")),
    ("logreg_predict.py", (TEST,)),
])
def test_cli_usage(run, script, args):
    r = run(script, *args)
    assert r.returncode != 0 and "usage" in r.stderr


def test_cli_predict_bad_weights(run, write_csv):
    r = run("logreg_predict.py", TEST, write_csv("{not json", "w.json"))
    assert r.returncode != 0 and "cannot load weights" in r.stderr


def test_cli_predict_missing_weights(run):
    r = run("logreg_predict.py", TEST, "nope.json")
    assert r.returncode != 0 and "cannot load weights" in r.stderr


def test_cli_predict_missing_feature_column(run, tmp_path, write_csv):
    assert run("logreg_train.py", TRAIN).returncode == 0
    r = run("logreg_predict.py", write_csv("Index,Astronomy\n0,1\n"), "weights.json")
    assert r.returncode != 0 and "missing feature columns" in r.stderr
