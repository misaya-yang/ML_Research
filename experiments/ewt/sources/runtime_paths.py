"""Put reproductions in a fresh directory, outside all frozen result folders."""
import os
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "experiments" / "ewt" / "data"
LOCAL_RUNS = (PROJECT / "results" / "local_runs").resolve()

def new_run_dir(prefix):
    parent = Path(os.environ.get("IMH_RUN_DIR", str(LOCAL_RUNS))).expanduser().resolve()
    if (parent == PROJECT or PROJECT in parent.parents) and not (
        parent == LOCAL_RUNS or LOCAL_RUNS in parent.parents
    ):
        raise ValueError("Within this project, run output must be under results/local_runs")
    parent.mkdir(parents=True, exist_ok=True)
    result = Path(tempfile.mkdtemp(prefix=prefix + "-", dir=parent))
    print(f"Run output: {result}", flush=True)
    return result
