"""Generate experimental food highlights for arbitrary uploaded photos in memory."""
import base64
from functools import lru_cache
import io
import json
import numpy as np
from PIL import Image, ImageOps
from .inference import ROOT, ENCODER_LOCK, decode_image
from .input_checks import pixel_digest
from .segmentation import load_segmenter

@lru_cache(maxsize=1)
def segmenter(root=ROOT):
    try:
        return load_segmenter(root)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        raise RuntimeError('Food highlighting is unavailable. Install requirements-segmentation.txt and the pinned weights described in docs/EXPERIMENTS.md, then restart the demo.') from exc

def highlight_photos(before, after, root=ROOT):
    photos = [decode_image(before), decode_image(after)]
    if any((min(photo.size) < 224 for photo in photos)):
        raise ValueError('Highlighting needs photos at least 224 pixels wide and tall.')
    with ENCODER_LOCK:
        model, transform, torch = segmenter(root)
        with torch.inference_mode():
            masks = [(model(transform(photo)[None])[0].argmax(dim=0) != 0).cpu().numpy() for photo in photos]
    images = {}
    for role, photo, mask in zip(('before', 'after'), photos, masks):
        thumbnail = ImageOps.contain(photo, (1000, 1000))
        alpha = Image.fromarray(mask.astype(np.uint8) * 255).resize(thumbnail.size, Image.Resampling.NEAREST).point(lambda v: int(v * 0.35))
        rendered = Image.composite(Image.new('RGB', thumbnail.size, (0, 140, 255)), thumbnail, alpha)
        output = io.BytesIO()
        rendered.save(output, format='JPEG', quality=90)
        images[role] = 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode()
    spec = json.loads((root / 'docs/SEGMENTATION-PROTOCOL.json').read_text())['model']
    return {'status': 'highlighted', 'images': images, 'model_revision': spec['revision'], 'label_type': 'experimental_model_prediction_not_ground_truth', 'affects_estimated_fraction': False, 'decoded_image_sha256': {role: pixel_digest(photo) for role, photo in zip(('before', 'after'), photos)}}
