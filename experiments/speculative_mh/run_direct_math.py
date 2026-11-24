"""Direct task-level alternatives on the same MATH rows, model, dtype and prompt as run_spec_math.py:
independent q_alpha samples (temperature 1/alpha, the PowerMH proposal law) from one cached prompt
prefill. Analysis turns them into a single low-temperature answer and anytime majority votes.

python run_direct_math.py --config configs/math10.json --samples 8 --approve
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from math_contract import boxed, norm, prompt_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--approve", action="store_true")
    a = ap.parse_args()
    if not a.approve:
        sys.exit("refusing a real-model run without --approve")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prefix_decoder import Decoder
    cfg = json.loads(Path(a.config).read_text())
    out = (HERE / cfg["run_dir"]).resolve()
    rows = [json.loads(l) for l in Path(cfg["data"]).read_text().splitlines() if l.strip()]
    lo, hi = map(int, cfg["rows"].split(":"))
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], dtype=getattr(torch, cfg["dtype"]),
                                                 attn_implementation="sdpa").to(cfg["device"]).eval()
    eos = {tok.convert_tokens_to_ids(s) for s in ("<|endoftext|>", "<|im_end|>")}
    with torch.inference_mode():
        for i in range(lo, hi):
            path = out / f"direct_row{i:04d}_qalpha.json"
            if path.exists():
                continue
            gc.collect()
            row = rows[i]
            ids = prompt_ids(tok, row)
            t0 = time.perf_counter()
            dec = Decoder(model, ids, cfg["alpha"], 1, cfg["device"])
            samples = []
            for s in range(a.samples):
                ts = time.perf_counter()
                inputs0 = dec.stats["decode_inputs"]
                node, gen = dec.root, []
                while True:
                    v = node.sample(np.random.default_rng([cfg["seed"] + 1, i, s, len(gen)]).random())
                    gen.append(v)
                    if v in eos or len(gen) == cfg["T"]:
                        break
                    node = dec.children([(node, v)], [f"sample{s}"])[0]
                ans = boxed(tok.decode(gen, skip_special_tokens=True))
                samples.append(dict(sample=s, n_tokens=len(gen), answer=ans, norm=norm(ans),
                                    correct=norm(ans) == norm(row["answer"]),
                                    decode_inputs=dec.stats["decode_inputs"] - inputs0, wall_s=time.perf_counter() - ts))
            rec = dict(row=i, gold=row["answer"], prompt_len=len(ids), prefill_s=dec.stats["prefill_s"],
                       total_wall_s=time.perf_counter() - t0, samples=samples, stats=dec.stats)
            path.write_text(json.dumps(rec))
            print(f"direct row {i} wall {rec['total_wall_s']:.1f} correct {[x['correct'] for x in samples]}", flush=True)


if __name__ == "__main__":
    main()
