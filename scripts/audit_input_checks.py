"""Audit deterministic checks on source images; not a food-classifier evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foodvision.inference import ROOT, decode_image
from foodvision.input_checks import inspect_pair, load_flags, POLICY_VERSION

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.research_root.resolve()):
        raise ValueError('Choose a new output outside the original research workspace.')
    rows = json.loads((args.research_root / 'results/lefood_records.json').read_text())
    hashes = json.loads((args.research_root / 'results/image_hashes.json').read_text())
    paths_list = list((args.research_root / 'lefood').rglob('*.JPG'))
    paths = {p.name: p for p in paths_list}
    if len(paths) != len(paths_list):
        raise ValueError('Ambiguous image names')
    flags = load_flags(ROOT)
    results = []
    for row in rows:
        images = []
        for role in ('before', 'after'):
            name = row[role + '_image']
            raw = paths[name].read_bytes()
            if hashlib.sha256(raw).hexdigest() != hashes[name]:
                raise ValueError('Source image changed')
            images.append(decode_image(raw))
        checks = inspect_pair(images, 'visible_food', flags)
        results.append({'record_id': f"L{row['source_id']}", 'status': checks['status'], 'issues': checks['issues']})
    report = {'policy_version': POLICY_VERSION, 'scope': 'Structural/file audit only. visible_food is injected to isolate automated checks, not asserted as ground truth. No empty-plate sensitivity or specificity is measured.', 'records': len(results), 'status_counts': dict(Counter((r['status'] for r in results))), 'issue_counts': dict(Counter((i['code'] for r in results for i in r['issues']))), 'rows': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
if __name__ == '__main__':
    main()
