"""Frozen relative-depth experiment; the outputs are not metric measurements."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from .benchmark import ROOT, verify_payload
FEATURE_NAMES = ('food_area', 'reference_available', 'median_residual', 'q90_residual', 'residual_std', 'positive_residual_integral', 'positive_residual_fraction', 'reference_residual_std', 'depth_std')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def shape_features(depth, mask):
    """Affine-depth-invariant shape cues with an unverified local background ring."""
    depth, mask = (np.asarray(depth, dtype=float), np.asarray(mask, dtype=bool))
    if depth.ndim != 2 or depth.shape != mask.shape or (not depth.size) or (not np.isfinite(depth).all()):
        raise ValueError('Expected aligned finite nonempty 2D depth and mask arrays.')
    low, high = np.quantile(depth, [0.05, 0.95])
    normalized = (depth - low) / (high - low) if high - low > 1e-10 else np.zeros_like(depth)
    result = np.zeros(len(FEATURE_NAMES))
    result[0], result[-1] = (mask.mean(), normalized.std())
    ring = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(31))) > 0
    ring &= ~mask
    if not mask.any() or ring.sum() < 32:
        return result
    y, x = np.meshgrid(np.linspace(-1, 1, depth.shape[0]), np.linspace(-1, 1, depth.shape[1]), indexing='ij')
    plane = np.stack([np.ones_like(x), x, y], axis=-1)
    coefficients = np.linalg.lstsq(plane[ring], normalized[ring], rcond=None)[0]
    residual = normalized - plane @ coefficients
    food = residual[mask]
    result[1:8] = [1, np.median(food), np.quantile(food, 0.9), food.std(), np.maximum(food, 0).sum() / depth.size, (food > 1e-06).mean(), residual[ring].std()]
    return result

def pair_features(before, after):
    before, after = (np.asarray(before, dtype=float), np.asarray(after, dtype=float))
    if before.ndim != 2 or before.shape != after.shape or before.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Expected matching depth feature matrices.')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Depth features must be finite.')
    return np.column_stack([before, after, after - before])

def load_depth(root=ROOT, device='cpu'):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    spec = json.loads((root / 'docs/DEPTH-PROTOCOL.json').read_text())['model']
    directory = root / 'models/depth-cache/small-v2'
    for name, expected in spec['files'].items():
        if not (directory / name).is_file() or digest(directory / name) != expected:
            raise ValueError(f'Missing or changed pinned depth asset: {name}')
    torch.set_num_threads(4)
    processor = AutoImageProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(directory, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval().to(device)
    return (model, processor, torch)

def infer(photo, model, processor, torch, device):
    with torch.inference_mode():
        output = model(**processor(images=photo, return_tensors='pt').to(device)).predicted_depth
        depth = torch.nn.functional.interpolate(output[:, None], size=(256, 256), mode='bilinear', align_corners=False)
    result = depth[0, 0].cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError('Depth inference produced non-finite values.')
    return result

def extract(research_root, mask_cache, output, device='cpu', root=ROOT):
    research_root, mask_cache, output = map(Path, (research_root, mask_cache, output))
    if output.exists() or output.resolve().is_relative_to(research_root.resolve()):
        raise ValueError('Choose a new cache outside the original research workspace.')
    verify_payload(root)
    protocol_path = root / 'docs/DEPTH-PROTOCOL.json'
    mask_manifest = json.loads((mask_cache / 'manifest.json').read_text())
    records = [r for r in json.loads((root / 'data/records.json').read_text()) if r['source'] == 'LeFood']
    if [r['record_id'] for r in records] != [r['record_id'] for r in mask_manifest['records']]:
        raise ValueError('Mask provenance and packaged records differ.')
    paths = list((research_root / 'lefood').rglob('*.JPG'))
    files = {p.name: p for p in paths}
    if len(files) != len(paths):
        raise ValueError('Ambiguous image filenames.')
    model, processor, torch = load_depth(root, device)
    output.mkdir(parents=True)
    (output / 'maps').mkdir()
    started = time.monotonic()
    arrays, provenance = ({'before': [], 'after': []}, [])
    for i, row in enumerate(mask_manifest['records']):
        item = {'record_id': row['record_id']}
        for role in arrays:
            source = row[role]
            path = files[source['filename']]
            if digest(path) != source['sha256']:
                raise ValueError(f"Source image changed: {source['filename']}")
            with Image.open(path) as image:
                photo = ImageOps.exif_transpose(image).convert('RGB')
            mask_path = mask_cache / source['mask']
            with Image.open(mask_path) as image:
                mask = np.asarray(image.resize((256, 256), Image.Resampling.NEAREST)) > 0
            depth = infer(photo, model, processor, torch, device)
            arrays[role].append(shape_features(depth, mask))
            map_path = output / 'maps' / f"{row['record_id']}_{role}.npz"
            np.savez_compressed(map_path, inverse_depth=depth)
            item[role] = {'filename': source['filename'], 'sha256': source['sha256'], 'mask_sha256': digest(mask_path), 'depth_map': str(map_path.relative_to(output)), 'depth_sha256': digest(map_path)}
        provenance.append(item)
        if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(records):
            print(f'Depth {i + 1}/{len(records)} pairs in {time.monotonic() - started:.1f}s ({device})', flush=True)
    before, after = (np.asarray(arrays['before']), np.asarray(arrays['after']))
    np.savez_compressed(output / 'features.npz', record_ids=[r['record_id'] for r in records], before=before, after=after, paired=pair_features(before, after))
    from importlib.metadata import version
    manifest = {'status': 'complete', 'measurement': 'predicted_relative_inverse_depth_not_metric_volume', 'protocol_sha256': digest(protocol_path), 'extractor_sha256': digest(Path(__file__)), 'mask_manifest_sha256': digest(mask_cache / 'manifest.json'), 'features_sha256': digest(output / 'features.npz'), 'device': device, 'model': json.loads(protocol_path.read_text())['model'], 'feature_names': FEATURE_NAMES, 'elapsed_seconds': time.monotonic() - started, 'records': provenance, 'environment': {p: version(p) for p in ('torch', 'transformers', 'numpy', 'Pillow')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--mask-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = parser.parse_args()
    extract(args.research_root, args.mask_cache, args.output, args.device)
