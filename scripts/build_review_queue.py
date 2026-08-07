"""Join saved AI proposals to source evidence; never create corrected mass labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

def assemble(proposals, sources, evidence):
    if proposals['human_review'] != 'pending' or proposals['evaluation_gold'] or proposals['training_eligible']:
        raise ValueError('This import supports unverified AI proposals only.')
    labels = []
    for page in range(1, (len(sources) + 23) // 24 + 1):
        codes = proposals['pages'][str(page)].split()
        if len(codes) != min(24, len(sources) - len(labels)) or set(codes) - {'F', 'R', 'U'}:
            raise ValueError('Contact-sheet codes must cover every cell exactly once.')
        labels.extend(codes)
    if set(proposals['pages']) != {str(i) for i in range(1, (len(sources) + 23) // 24 + 1)}:
        raise ValueError('Unexpected contact-sheet pages.')
    expected_ids = [f"L{r['source_id']}" for r in sources]
    if len(set(expected_ids)) != len(expected_ids) or set(expected_ids) != set(evidence):
        raise ValueError('Source IDs and image provenance must match exactly.')
    if set(proposals['before_overrides']) - set(expected_ids):
        raise ValueError('Unknown before-image override.')
    rows = []
    for i, (source, after) in enumerate(zip(sources, labels)):
        record_id = expected_ids[i]
        before = proposals['before_overrides'].get(record_id, 'F')
        if before not in {'F', 'R', 'U'}:
            raise ValueError('Unknown before-image code.')
        flags = []
        if after == 'F' and source['after_g'] == 0:
            flags.append('visible_material_with_recorded_zero_mass')
        if before != 'F' and source['before_g'] > 0:
            flags.append('starting_photo_unclear_despite_positive_mass')
        if after == 'U':
            flags.append('uncertain_material')
        if after == 'R' and source['fraction'] > 0.1:
            flags.append('no_clear_portion_despite_recorded_fraction_above_10_percent')
        note = proposals.get('notes', {}).get(record_id)
        if note:
            flags.append('reviewer_note')
        images = evidence[record_id]
        if any((images[k]['filename'] != source[f'{k}_image'] for k in ('before', 'after'))):
            raise ValueError('Source and reviewed-image filenames differ.')
        rows.append({'record_id': record_id, 'page': i // 24 + 1, 'cell': i % 24 + 1, 'before_visual_proposal': before, 'after_visual_proposal': after, 'human_review': 'pending', 'evaluation_gold': False, 'training_eligible': False, 'flags': flags, 'note': note, 'source_excel_row': source['excel_row'], 'recorded_before_g': source['before_g'], 'recorded_after_g': source['after_g'], 'recorded_fraction': source['fraction'], 'images': {k: {f: images[k][f] for f in ('filename', 'sha256')} for k in ('before', 'after')}})
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--proposals', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.research_root.resolve()):
        raise ValueError('Choose a new directory outside the original research workspace.')
    raw = args.proposals.read_bytes()
    proposals = json.loads(raw)
    sources = json.loads((args.research_root / 'results/lefood_records.json').read_text())
    evidence = {r['record_id']: r for r in json.loads(args.manifest.read_text())['records']}
    rows = assemble(proposals, sources, evidence)
    source_paths = list((args.research_root / 'lefood').rglob('*.JPG'))
    paths = {p.name: p for p in source_paths}
    if len(paths) != len(source_paths):
        raise ValueError('Ambiguous image basenames.')
    for row in rows:
        for item in row['images'].values():
            if hashlib.sha256(paths[item['filename']].read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Source image changed since visual review.')
    sheets = {f'page-{i:02}.jpg': hashlib.sha256((args.proposals.parent / f'page-{i:02}.jpg').read_bytes()).hexdigest() for i in range(1, (len(rows) + 23) // 24 + 1)}
    report = {'provenance': {k: v for k, v in proposals.items() if k not in ('pages', 'notes', 'before_overrides')}, 'proposal_sha256': hashlib.sha256(raw).hexdigest(), 'review_sheet_sha256': sheets, 'limitations': 'Thumbnail AI screening, not independent human verification. Flags are review priorities, not confirmed errors. R does not mean zero grams. All rows remain in the mass benchmark.', 'priority_rule': 'Join visual proposals with unchanged source weights after screening: F with zero after mass, unclear before with positive mass, U material, R with fraction >10%, or explicit reviewer note. Not a classifier operating threshold.', 'summary': {'pairs': len(rows), 'after_codes': dict(Counter((r['after_visual_proposal'] for r in rows))), 'flagged_pairs': sum((bool(r['flags']) for r in rows)), 'flag_counts': dict(Counter((f for r in rows for f in r['flags'])))}, 'rows': rows}
    args.output.mkdir(parents=True)
    (args.output / 'review-queue.json').write_text(json.dumps(report, indent=2) + '\n')
    selected = [r for r in rows if any((f not in ('uncertain_material', 'reviewer_note') for f in r['flags']))]
    for page, start in enumerate(range(0, len(selected), 8), 1):
        canvas = Image.new('RGB', (1000, 270 * min(8, len(selected) - start)), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(selected[start:start + 8]):
            y = j * 270
            draw.text((10, y + 8), f"{row['record_id']} | recorded {row['recorded_before_g']}g -> {row['recorded_after_g']}g | AI proposal; human review pending", fill='black')
            for k, kind in enumerate(('before', 'after')):
                with Image.open(paths[row['images'][kind]['filename']]) as im:
                    photo = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (490, 230))
                canvas.paste(photo, (k * 500 + (500 - photo.width) // 2, y + 30))
        canvas.save(args.output / f'source-conflict-proposals-{page:02}.jpg', quality=90)
    print(json.dumps(report['summary'], indent=2))
if __name__ == '__main__':
    main()
