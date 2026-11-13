import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "datasets" / "dataset_train.csv"
TEST = ROOT / "datasets" / "dataset_test.csv"


@pytest.fixture
def run(tmp_path):
    """Run a project script with tmp_path as cwd; returns CompletedProcess."""
    def _run(script, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            cwd=tmp_path, capture_output=True, text=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
    return _run


@pytest.fixture
def write_csv(tmp_path):
    def _write(text, name="data.csv"):
        p = tmp_path / name
        p.write_text(text)
        return p
    return _write
