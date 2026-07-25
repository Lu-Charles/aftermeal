"""Optional frozen food-mask features. Predicted masks are never gold labels."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
from .benchmark import ROOT, verify_payload
FEATURE_NAMES = ('food_area_fraction', 'food_probability_mean', 'food_probability_std', 'foreground_bbox_area_fraction', 'foreground_centroid_x', 'foreground_centroid_y')

def mask_features(mask, food_probability):
    mask = np.asarray(mask, dtype=bool)
    probability = np.asarray(food_probability, dtype=float)
    if mask.ndim != 2 or probability.shape != mask.shape or (not mask.size):
        raise ValueError('Expected aligned nonempty two-dimensional masks and probabilities.')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Food probabilities must be finite values between zero and one.')
    y, x = np.nonzero(mask)
    height, width = mask.shape
    bbox = (x.max() - x.min() + 1) * (y.max() - y.min() + 1) / mask.size if len(x) else 0
    return np.array([mask.mean(), probability.mean(), probability.std(), bbox, (x.mean() + 0.5) / width if len(x) else 0.5, (y.mean() + 0.5) / height if len(y) else 0.5], dtype=np.float64)

def pair_features(before, after):
    before, after = (np.asarray(before, dtype=float), np.asarray(after, dtype=float))
    if before.ndim != 2 or before.shape != after.shape or before.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Expected matching before/after mask feature matrices.')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Mask features must be finite.')
    ratio = np.clip(after[:, 0] / np.maximum(before[:, 0], 1 / 512 ** 2), 0, 2)
    return np.column_stack([before, after, after - before, ratio])

def load_segmenter(root=ROOT):
    import torch
    import segmentation_models_pytorch as smp
    from torchvision import transforms
    protocol = json.loads((root / 'docs/SEGMENTATION-PROTOCOL.json').read_text())
    spec = protocol['model']
    weights = root / 'models/segmentation-cache' / spec['weights']
    if not weights.exists() or hashlib.sha256(weights.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Download the pinned segmentation weights and verify their checksum first.')
    torch.set_num_threads(4)
    model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', encoder_weights=None, in_channels=3, classes=104)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.eval().to('cpu')
    transform = transforms.Compose([transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return (model, transform, torch)

def extract(research_root: Path, output: Path, root: Path=ROOT):
    if output.exists() or output.resolve().is_relative_to(research_root.resolve()):
        raise ValueError('Choose a new cache directory outside the original research workspace.')
    verify_payload(root)
    records = json.loads((root / 'data/records.json').read_text())
    lefood = [r for r in records if r['source'] == 'LeFood']
    original = {f"L{r['source_id']}": r for r in json.loads((research_root / 'results/lefood_records.json').read_text())}
    hashes = json.loads((research_root / 'results/image_hashes.json').read_text())
    paths = list((research_root / 'lefood').rglob('*.JPG'))
    files = {p.name: p for p in paths}
    if len(files) != len(paths):
        raise ValueError('Image basenames are ambiguous.')
    model, transform, torch = load_segmenter(root)
    output.mkdir(parents=True)
    (output / 'masks').mkdir()
    before, after, provenance = ([], [], [])
    started = time.monotonic()
    with torch.inference_mode():
        for i, record in enumerate(lefood):
            source = original[record['record_id']]
            if abs(record['fraction'] - source['fraction']) > 1e-09:
                raise ValueError('Source mass ratios differ from the packaged records.')
            item = {'record_id': record['record_id']}
            for kind, features in (('before', before), ('after', after)):
                filename = source[f'{kind}_image']
                path = files[filename]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != hashes[filename]:
                    raise ValueError(f'Image bytes differ from original extraction: {filename}')
                with Image.open(path) as image:
                    photo = ImageOps.exif_transpose(image).convert('RGB')
                logits = model(transform(photo)[None])[0]
                mask = (logits.argmax(dim=0) != 0).numpy()
                probability = (1 - logits.softmax(dim=0)[0]).numpy()
                features.append(mask_features(mask, probability))
                mask_name = f"{record['record_id']}_{kind}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'masks' / mask_name)
                item[kind] = {'filename': filename, 'sha256': digest, 'mask': 'masks/' + mask_name}
            provenance.append(item)
            if (i + 1) % 25 == 0 or i + 1 == len(lefood):
                print(f'Segmented {i + 1}/{len(lefood)} pairs in {time.monotonic() - started:.1f}s', flush=True)
    before, after = (np.array(before), np.array(after))
    np.savez_compressed(output / 'features.npz', record_ids=[r['record_id'] for r in lefood], before=before, after=after, paired=pair_features(before, after))
    from importlib.metadata import version
    metadata = {'status': 'complete', 'label_type': 'model_prediction_not_ground_truth', 'device': 'cpu', 'protocol_sha256': hashlib.sha256((root / 'docs/SEGMENTATION-PROTOCOL.json').read_bytes()).hexdigest(), 'features_sha256': hashlib.sha256((output / 'features.npz').read_bytes()).hexdigest(), 'feature_names': FEATURE_NAMES, 'model': json.loads((root / 'docs/SEGMENTATION-PROTOCOL.json').read_text())['model'], 'environment': {p: version(p) for p in ('torch', 'torchvision', 'segmentation-models-pytorch', 'timm', 'numpy', 'Pillow')}, 'elapsed_seconds': time.monotonic() - started, 'records': provenance}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
