"""Native run on PowerMH's own workload: MATH-500 rows with Qwen2.5-Math-1.5B (plain completion
prompt of the repository's native PowerMH runs), T=512, K=16, alpha=4. Same engine and tape addressing
as run_spec.py; every arm of a problem shares one tape. Delivery is the final state (native contract);
the whole trajectory is compared pathwise, so any state-level verifier's first-hit ledger is preserved.

python run_spec_math.py --config configs/math10.json --approve
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from math_contract import boxed, norm, prompt_ids  # noqa: E402
from spec_engine import Engine, parity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--approve", action="store_true")
    a = ap.parse_args()
    if not a.approve:
        sys.exit("refusing a real-model run without --approve")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.loads(Path(a.config).read_text())
    out = (HERE / cfg["run_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(cfg["data"]).read_text().splitlines() if l.strip()]
    lo, hi = map(int, cfg["rows"].split(":"))
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], dtype=getattr(torch, cfg["dtype"]),
                                                 attn_implementation="sdpa").to(cfg["device"]).eval()
    eos = sorted({tok.convert_tokens_to_ids(s) for s in ("<|endoftext|>", "<|im_end|>")})
    runs = [tuple(x) for x in cfg["arms"]]
    ref = tuple(cfg["parity_reference"])
    with torch.inference_mode():
        for i in range(lo, hi):
            row = rows[i]
            ids = prompt_ids(tok, row)
            order = runs[i % len(runs):] + runs[:i % len(runs)]
            recs = {}
            for mode, shape in order:
                path = out / f"row{i:04d}_{mode}_{shape}.json"
                if path.exists():
                    recs[mode, shape] = json.loads(path.read_text())
                    continue
                gc.collect()
                t0 = time.perf_counter()
                rec = Engine(model, ids, eos, cfg, i, mode, shape=shape, device=cfg["device"]).execute()
                rec["total_wall_s"] = time.perf_counter() - t0
                ans = boxed(tok.decode(rec["final_ids"], skip_special_tokens=True))
                rec.update(row=i, unique_id=row.get("unique_id"), gold=row["answer"], answer=ans,
                           correct=norm(ans) == norm(row["answer"]), prompt_len=len(ids),
                           prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                tmp = path.with_suffix(".partial")
                tmp.write_text(json.dumps(rec))
                tmp.replace(path)
                recs[mode, shape] = rec
                print(f"math row {i} {mode}/{shape} ticks {rec['ticks']} calls {rec['decode_calls']} "
                      f"live {rec['live_unique_decode_inputs']} dummy {rec['dummy_decode_inputs']} "
                      f"wall {rec['total_wall_s']:.1f} answer {ans!r} correct {rec['correct']}", flush=True)
            with open(out / "parity.jsonl", "a") as f:
                for k, rec in recs.items():
                    if k != ref:
                        f.write(json.dumps(dict(row=i, mode=k[0], shape=k[1], **parity(recs[ref], rec))) + "\n")


if __name__ == "__main__":
    main()
