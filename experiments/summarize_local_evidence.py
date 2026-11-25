"""Extract a small receipt from known local run JSONs; never scans caches or runs models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    'pgcd_cpu': 'pgcd-finite-policy-20261124/cpu_confirmation_complete/pgcd_work/confirm16_cpu_impact_resume.combined_summary.json',
    'pgcd_m4': 'pgcd-finite-policy-20261124/confirm16_mps_impact.summary.json',
    'math_cap': 'math-cap-continuation-20261124/offline_scores.json',
}


def extract(local):
    out = {}
    for key, relative in SOURCES.items():
        data = (local / relative).read_bytes()
        obj = json.loads(data)
        if key.startswith('pgcd'):
            checks = obj['coupling_checks']
            item = {
                'aggregate': obj['aggregate'],
                'matched_same_device_checks': sum(bool(x['matched']) for x in checks),
                'total_same_device_checks': len(checks),
            }
        else:
            keep = ('frozen_inputs', 'original_status_counts',
                    'eligible_cap_abstentions', 'eligible_final_status_counts',
                    'strict_eligible_endpoint_correct')
            item = {'scope': obj['scope'], 'limitations': obj['limitations'],
                    'arms': {arm: {k: value[k] for k in keep}
                             for arm, value in obj['arms'].items()}}
        out[key] = {'local_source': relative,
                    'sha256': hashlib.sha256(data).hexdigest(), **item}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-runs', type=Path, default=ROOT / 'results/local_runs')
    p.add_argument('--compare', type=Path)
    args = p.parse_args()
    result = extract(args.local_runs)
    if args.compare and result != json.loads(args.compare.read_text()):
        p.exit(1, 'Compact evidence or source hashes differ.\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
