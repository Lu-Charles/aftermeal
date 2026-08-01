"""Render unchanged predicted masks on explicit diagnostic examples."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=['L133', 'L81', 'L82', 'L400', 'L117', 'L476'])
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.research_root.resolve()):
        raise ValueError('Choose a new output outside the original research workspace.')
    rows = {r['record_id']: r for r in json.loads((args.cache / 'manifest.json').read_text())['records']}
    paths = {p.name: p for p in (args.research_root / 'lefood').rglob('*.JPG')}
    canvas = Image.new('RGB', (1000, 290 * len(args.ids) + 40), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), 'Frozen model predictions: blue = predicted food. Diagnostic examples; no mask edits. Not validated masks.', fill='black')
    for j, record_id in enumerate(args.ids):
        y = 40 + j * 290
        draw.text((10, y), f'{record_id} before (left) / after (right)', fill='black')
        for k, kind in enumerate(('before', 'after')):
            item = rows[record_id][kind]
            path = paths[item['filename']]
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Raw image differs from segmentation input.')
            with Image.open(path) as im:
                photo = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (490, 260))
            with Image.open(args.cache / item['mask']) as im:
                mask = im.convert('L').resize(photo.size, Image.Resampling.NEAREST).point(lambda v: int(v * 0.35))
            overlay = Image.new('RGB', photo.size, (0, 140, 255))
            photo = Image.composite(overlay, photo, mask)
            canvas.paste(photo, (k * 500 + (500 - photo.width) // 2, y + 20))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
if __name__ == '__main__':
    main()
