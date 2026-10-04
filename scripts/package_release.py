"""Create a standalone source ZIP, excluding local records, environments and weights."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
ROOT = Path(__file__).resolve().parent.parent

def build(output):
    if output.exists():
        raise FileExistsError(f'Choose a new release path: {output}')
    paths = set()
    patterns = ['README.md', 'DESIGN.md', 'LICENSE', '.gitignore', '.dockerignore', 'Dockerfile', 'gunicorn.conf.py', 'requirements*.txt', '.github/workflows/*.yml', 'foodvision/*.py', 'scripts/*.py', 'tests/*.py', 'tests/*.mjs', 'web/*.html', 'web/*.css', 'web/*.js', 'web/*.mjs', 'docs/*.md', 'docs/*.json', 'docs/*.png', 'docs/*.jpg', 'data/*.json', 'data/features.npz', 'models/demo_model.*', 'examples/*.json', 'examples/*.npz', 'examples/images/*.jpg', 'examples/images/overlays/*.jpg', 'reports/ai-review/page-01.jpg', 'reports/ai-review/page-02.jpg', 'reports/ai-review/page-03.jpg', 'reports/ai-review/page-04.jpg', 'reports/ai-review/page-05.jpg', 'reports/ai-review/page-06.jpg', 'reports/ai-review/page-07.jpg', 'reports/ai-review/page-08.jpg', 'reports/ai-review/page-09.jpg', 'reports/ai-review/page-10.jpg', 'reports/ai-review/page-11.jpg', 'reports/ai-review/page-12.jpg', 'reports/ai-review/page-13.jpg', 'reports/ai-review/page-14.jpg', 'reports/ai-review/page-15.jpg', 'reports/ai-review/page-16.jpg', 'reports/ai-review/page-17.jpg', 'reports/ai-review/page-18.jpg', 'reports/ai-review/page-19.jpg', 'reports/ai-review/page-20.jpg', 'reports/ai-review/page-21.jpg', 'reports/ai-review/page-22.jpg', 'reports/ai-review/page-proposals.json', 'reports/baseline-recheck.json', 'reports/benchmark.json', 'reports/data-budget.json', 'reports/depth-blend.json', 'reports/depth-diagnostics.png', 'reports/depth-experiment.json', 'reports/diagnostics.json', 'reports/hurdle.json', 'reports/input-checks.json', 'reports/new-photo-recheck.json', 'reports/review-queue/review-queue.json', 'reports/review-queue/source-conflict-proposals-01.jpg', 'reports/review-queue/source-conflict-proposals-02.jpg', 'reports/segmentation-experiment.json', 'reports/segmentation-preview.jpg', 'reports/source-audit/review-pairs.jpg', 'reports/source-audit/source-audit.json']
    for pattern in patterns:
        paths.update(ROOT.glob(pattern))
    entries, redactions = ({}, [])
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Unexpected release asset: {path}')
        name = path.relative_to(ROOT).as_posix()
        if name == 'docs/capture-mobile-test.png':
            continue
        raw = path.read_bytes()
        if name == 'reports/source-audit/source-audit.json':
            data = json.loads(raw)
            data['workbook'] = 'LeFood-Set/data_original.xlsx'
            raw = (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode()
            redactions.append({'file': name, 'field': 'workbook', 'change': 'Absolute local path replaced with dataset-relative description; numerical evidence unchanged.'})
        if path.suffix in ('.py', '.json', '.md', '.html', '.js', '.mjs', '.yml', '.txt'):
            if re.search(b'/(?:Users|home)/[A-Za-z0-9_.-]+/', raw):
                raise ValueError(f'Local path needs review before release: {name}')
        entries[name] = raw
    manifest = {'version': '1.1.0', 'kind': 'standalone_source_release', 'redactions': redactions, 'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in entries.items()}}
    entries['RELEASE-MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', compression=ZIP_DEFLATED) as archive:
        for name, raw in entries.items():
            info = ZipInfo('aftermeal/' + name, date_time=datetime.now(timezone.utc).timetuple()[:6])
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 33188 << 16
            archive.writestr(info, raw)
    print(f'Packaged {len(entries)} files: {output} ({output.stat().st_size / 1024 ** 2:.1f} MiB)')
    print('SHA256 ' + hashlib.sha256(output.read_bytes()).hexdigest())
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    build(parser.parse_args().output)
