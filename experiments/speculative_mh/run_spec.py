"""Native run: serial and depth-two speculative PowerMH on a frozen MBPP+ panel, plus direct retries.

python run_spec.py --config configs/fresh40.json --stage mh|direct [--limit N] --approve

Every arm of one task uses the same tape. Per run we save the full step ledger (with the tick, call
count, wall time and work counters at which each decision became known), the model-call trace
(width, live rows, forward seconds) and the decoded program of every state. Hidden tests are never
read here; grading is a separate offline step (grade_spec.py).
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "first_answer"))
from common import clean, file_sha256, load_panel  # noqa: E402
from spec_engine import Engine, parity  # noqa: E402


def runs_for(cfg):
    out = [(m, "compact") for m in cfg["serial_modes"]]
    for shape in cfg["tree_shapes"]:
        out += [(m, shape) for m in cfg["tree_modes"]]
    return out


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], dtype=getattr(torch, cfg["dtype"]),
                                                 attn_implementation="sdpa").to(cfg["device"]).eval()
    eos = sorted({tok.convert_tokens_to_ids(s) for s in ("<|endoftext|>", "<|im_end|>")})
    return tok, model, eos


def prompt_ids(tok, task):
    text = tok.apply_chat_template(task.messages(), add_generation_prompt=True, tokenize=False)
    ids = list(tok(text, add_special_tokens=False)["input_ids"])
    for frag in task.hidden_fragments():
        if frag in text:
            raise RuntimeError(f"hidden test fragment in prompt for task {task.task_id}")
    return ids


def program(tok, ids):
    text = tok.decode(ids, skip_special_tokens=True)
    return text, clean(text)


def run_mh(cfg, tok, model, eos, tasks, out, limit):
    import torch
    runs = runs_for(cfg)
    with torch.inference_mode():
        for i, task in enumerate(tasks[:limit]):
            ids = prompt_ids(tok, task)
            order = runs[i % len(runs):] + runs[:i % len(runs)]
            recs = {}
            for mode, shape in order:
                path = out / f"task{task.task_id}_{mode}_{shape}.json"
                if path.exists():
                    recs[mode, shape] = json.loads(path.read_text())
                    continue
                gc.collect()
                t0 = time.perf_counter()
                eng = Engine(model, ids, eos, cfg, i, mode, shape=shape, device=cfg["device"])
                rec = eng.execute()
                rec["total_wall_s"] = time.perf_counter() - t0
                states = {}
                for s in rec["steps"]:
                    if s["post_ids"] is not None:
                        h = hashlib.sha256(json.dumps(s["post_ids"]).encode()).hexdigest()[:16]
                        s["state_hash"] = h
                        if h not in states:
                            text, code = program(tok, s["post_ids"])
                            states[h] = dict(text=text, code=code)
                rec.update(task_id=task.task_id, row=i, prompt_len=len(ids), states=states,
                           prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                tmp = path.with_suffix(".partial")
                tmp.write_text(json.dumps(rec))
                tmp.replace(path)
                recs[mode, shape] = rec
                print(f"mh task {task.task_id} {mode}/{shape} ticks {rec['ticks']} calls {rec['decode_calls']} "
                      f"live {rec['live_unique_decode_inputs']} dummy {rec['dummy_decode_inputs']} "
                      f"wall {rec['total_wall_s']:.2f}", flush=True)
                del eng
            ref = recs[("serial_eager", "compact")]
            with open(out / "parity.jsonl", "a") as f:
                for (mode, shape), rec in recs.items():
                    if (mode, shape) != ("serial_eager", "compact"):
                        f.write(json.dumps(dict(task_id=task.task_id, mode=mode, shape=shape, **parity(ref, rec))) + "\n")


def run_direct(cfg, tok, model, eos, tasks, out, limit):
    """Cached serial retries from one prompt prefill: greedy then T=1, and q_alpha (T=1/alpha) draws."""
    import torch
    from prefix_decoder import Decoder
    with torch.inference_mode():
        for i, task in enumerate(tasks[:limit]):
            ids = prompt_ids(tok, task)
            for pol in cfg["direct_policies"]:
                path = out / f"direct_task{task.task_id}_{pol['name']}.json"
                if path.exists():
                    continue
                gc.collect()
                t0 = time.perf_counter()
                dec = Decoder(model, ids, pol["alpha"], 1, cfg["device"])
                attempts = []
                for a in range(cfg["direct_attempts"]):
                    ta = time.perf_counter()
                    calls0, inputs0 = dec.stats["decode_calls"], dec.stats["decode_inputs"]
                    node, gen = dec.root, []
                    while True:
                        if pol["greedy_first"] and a == 0:
                            pmf = np.diff(np.r_[0.0, node.cdf])
                            v = int(node.ids[int(np.argmax(pmf))])
                        else:
                            u = np.random.default_rng([cfg["seed"] + 1, i, a, len(gen)]).random()
                            v = node.sample(u)
                        gen.append(v)
                        if v in eos or len(gen) == cfg["T"]:
                            break
                        node = dec.children([(node, v)], [f"attempt{a}"])[0]
                    text, code = program(tok, gen)
                    attempts.append(dict(attempt=a, ids=gen, text=text, code=code,
                                         decode_inputs=dec.stats["decode_inputs"] - inputs0,
                                         decode_calls=dec.stats["decode_calls"] - calls0,
                                         wall_s=time.perf_counter() - ta))
                rec = dict(task_id=task.task_id, row=i, policy=pol, prompt_len=len(ids),
                           prefill_s=dec.stats["prefill_s"],
                           total_wall_s=time.perf_counter() - t0, attempts=attempts, stats=dec.stats)
                path.write_text(json.dumps(rec))
                print(f"direct task {task.task_id} {pol['name']} wall {rec['total_wall_s']:.2f} "
                      f"inputs {dec.stats['decode_inputs']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=("mh", "direct"), required=True)
    ap.add_argument("--limit", type=int, default=10 ** 6)
    ap.add_argument("--out")
    ap.add_argument("--approve", action="store_true")
    a = ap.parse_args()
    if not a.approve:
        sys.exit("refusing a real-model run without --approve")
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out) if a.out else (HERE / cfg["run_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _, tasks = load_panel(HERE / cfg["panel"])
    prov = out / f"provenance_{a.stage}.json"
    if not prov.exists():
        prov.write_text(json.dumps(dict(config=cfg, config_sha256=file_sha256(a.config),
                                        panel_sha256=file_sha256(HERE / cfg["panel"]),
                                        engine_sha256=file_sha256(HERE / "spec_engine.py"),
                                        decoder_sha256=file_sha256(HERE / "prefix_decoder.py"),
                                        runner_sha256=file_sha256(__file__), started=time.ctime()), indent=1))
    import torch
    torch.set_num_threads(2)
    t = time.perf_counter()
    tok, model, eos = load_model(cfg)
    print("model loaded", round(time.perf_counter() - t, 1), "s; eos", eos, flush=True)
    (run_mh if a.stage == "mh" else run_direct)(cfg, tok, model, eos, tasks, out, a.limit)


if __name__ == "__main__":
    main()
