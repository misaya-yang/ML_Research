"""Check (or with --write, record) SHA-256 of every config, panel and code file.

Any edit after freezing makes verification fail; a real-model stage must not
start on a failed verification.
"""

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "configs/FROZEN.sha256"


def files():
    pats = ["*.py", "tests/*.py", "configs/*.json", "panels/*.json", "reproduce.sh"]
    return sorted(p for pat in pats for p in HERE.glob(pat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    cur = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
    if a.write:
        MANIFEST.write_text("".join(f"{h}  {p}\n" for p, h in cur.items()))
        print(f"wrote {len(cur)} hashes to {MANIFEST}")
        return
    if not MANIFEST.exists():
        sys.exit("no manifest; run with --write to freeze")
    old = {}
    for line in MANIFEST.read_text().splitlines():
        h, path = line.split("  ", 1)
        old[path.strip()] = h
    bad = sorted(k for k in set(old) | set(cur) if old.get(k) != cur.get(k))
    if bad:
        sys.exit("frozen verification FAILED for: " + ", ".join(bad))
    print(f"frozen verification ok ({len(cur)} files)")


if __name__ == "__main__":
    main()
