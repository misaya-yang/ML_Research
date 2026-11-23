"""Run a fixed, independently verified TabuCol-style baseline on official test graphs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("graphs", type=Path, help="NPZ with symmetric zero-diagonal 'adjacency' matrix")
    p.add_argument("out", type=Path, help="New output directory")
    args = p.parse_args()
    args.out.mkdir(exist_ok=False)
    src = Path(__file__).with_name("coloring_control.cpp")
    params = {"colors": 10, "max_moves": 20000, "max_seconds_per_graph": 0.1,
              "seed_by_row": "20261123 + row_id", "threads": 1,
              "compiler_flags": ["-std=c++17", "-O3"], "source_sha256": digest(src)}
    (args.out / "protocol.json").write_text(json.dumps(params, indent=2))
    clock = time.perf_counter()
    compiler = os.environ.get("CXX", "c++")
    binary = args.out.resolve() / "control"
    subprocess.run([compiler, *params["compiler_flags"], str(src), "-o", str(binary)], check=True)
    compile_s = time.perf_counter() - clock
    clock = time.perf_counter()
    with np.load(args.graphs, allow_pickle=False) as data:
        graph = data["adjacency"]
    if graph.shape != (1200, 200, 200):
        raise ValueError(f"Unexpected official test shape {graph.shape}")
    if (not np.isin(graph, [0, 1]).all() or graph.diagonal(axis1=1, axis2=2).any()
            or not np.array_equal(graph, graph.transpose(0, 2, 1))):
        raise ValueError("Expected unweighted undirected graphs with no loops")
    n = graph.shape[1]
    edge_path = args.out / "edges.txt"
    with edge_path.open("w") as stream:
        for idx, matrix in enumerate(graph):
            edges = np.transpose(np.nonzero(np.triu(matrix, k=1)))
            stream.write(f"{idx} {n} {len(edges)}\n")
            np.savetxt(stream, edges, fmt="%d %d")
    read_encode_s = time.perf_counter() - clock
    log = args.out / "outputs.txt"
    clock = time.perf_counter()
    with edge_path.open("r") as stdin, log.open("x") as stdout:
        subprocess.run([str(binary), "10", "20000", "0.1"], stdin=stdin, stdout=stdout, check=True)
    full_run_s = time.perf_counter() - clock
    clock = time.perf_counter()
    solved, times, min_conflicts = [], [], []
    for idx, row in enumerate(log.read_text().splitlines()):
        item, moves, seconds, best, *coloring = row.split()
        assert int(item) == idx and 0 <= int(moves) <= params["max_moves"]
        times.append(float(seconds))
        min_conflicts.append(int(best))
        if int(best) == 0:
            colors = np.array(coloring, dtype=np.int16)
            assert len(colors) == n and colors.min() >= 0 and colors.max() < params["colors"]
            assert not np.any(graph[idx] & (colors[:, None] == colors[None, :]))
            solved.append(idx)
        else:
            assert not coloring
    assert len(times) == len(graph)
    summary = {"instances": len(times), "verified_proper_colorings": len(solved),
               "abstained": len(times) - len(solved), "solved_ids": solved,
               "compile_s": compile_s, "read_encode_s": read_encode_s, "full_solver_process_s": full_run_s,
               "independent_verification_s": time.perf_counter() - clock,
               "sum_internal_solver_s": sum(times), "median_internal_solver_s": float(np.median(times)),
               "max_internal_solver_s": max(times), "min_conflicts_mean": float(np.mean(min_conflicts)),
               "graphs_sha256": digest(args.graphs), "output_sha256": digest(log), **params}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "solved_ids"}, indent=2))


if __name__ == "__main__":
    main()
