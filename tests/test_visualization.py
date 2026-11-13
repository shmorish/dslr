import pytest

from conftest import TEST, TRAIN
from dslr.plot import by_house

SCRIPTS = ["visualization/histogram.py", "visualization/scatter_plot.py", "visualization/pair_plot.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_saves_png(run, tmp_path, script):
    r = run(script, TRAIN, "out.png")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("script", SCRIPTS)
def test_rejects_unlabelled_dataset(run, script):
    r = run(script, TEST, "out.png")
    assert r.returncode != 0 and "no house labels" in r.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_rejects_dataset_without_house_column(run, write_csv, script):
    r = run(script, write_csv("Index,A\n0,1\n"), "out.png")
    assert r.returncode != 0 and "Hogwarts House" in r.stderr


@pytest.mark.parametrize("script", SCRIPTS)
def test_usage(run, script):
    assert "usage" in run(script).stderr


def test_scatter_rejects_missing_columns(run, write_csv):
    r = run("visualization/scatter_plot.py", write_csv("Hogwarts House,Astronomy\nSlytherin,1\n"), "o.png")
    assert r.returncode != 0 and "Defense Against the Dark Arts" in r.stderr
    assert "Traceback" not in r.stderr


def test_by_house_filters_house_and_missing():
    assert by_house([1, None, 3, 4], ["G", "G", "S", "G"], "G") == [1, 4]
