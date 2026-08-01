"""Package the existing frozen masks for the four demo samples, without retouching."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageOps
ROOT = Path(__file__).resolve().parents[1]

def export():
    cache = ROOT / 'data/segmentation-cache/mobilenet-v1'
    manifest_bytes = (cache / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    records = {row['record_id']: row for row in manifest['records']}
    raw = {p.name: p for p in (ROOT.parent / 'photo-waste-feasibility/lefood').rglob('*.JPG')}
    output = ROOT / 'examples/images/overlays'
    output.mkdir(parents=True, exist_ok=True)
    exported = []
    for sample in json.loads((ROOT / 'examples/examples.json').read_text()):
        for role in ('before', 'after'):
            source = records[sample['id']][role]
            path = raw[source['filename']]
            if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
                raise ValueError('Source photo differs from segmentation input.')
            mask_path = cache / source['mask']
            with Image.open(path) as photo_file:
                photo = ImageOps.contain(ImageOps.exif_transpose(photo_file).convert('RGB'), (1000, 1000))
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert('L').resize(photo.size, Image.Resampling.NEAREST).point(lambda v: int(v * 0.35))
            image = Image.composite(Image.new('RGB', photo.size, (0, 140, 255)), photo, mask)
            target = output / f"{sample['id']}_{role}.jpg"
            image.save(target, quality=90)
            exported.append({'record_id': sample['id'], 'role': role, 'source_sha256': source['sha256'], 'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest(), 'output': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    (ROOT / 'examples/overlay_manifest.json').write_text(json.dumps({'kind': 'frozen_model_prediction_not_ground_truth', 'model': manifest['model'], 'cache_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'rendering': 'Unedited argmax food mask, nearest-neighbor resize, blue RGB(0,140,255) at 35% opacity', 'affects_percentage_estimate': False, 'images': exported}, indent=2) + '\n')
    print(f'Exported {len(exported)} sample overlays.')
if __name__ == '__main__':
    export()
