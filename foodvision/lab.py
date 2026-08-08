"""Read-only experiment and annotation evidence for the local review workspace."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
from PIL import Image, ImageOps

class ReviewLab:

    def __init__(self, root):
        self.root = Path(root)
        self.research = self.root.parent / 'photo-waste-feasibility'

    @lru_cache(maxsize=1)
    def evidence(self):
        review = json.loads((self.root / 'reports/review-queue/review-queue.json').read_text())
        experiment = json.loads((self.root / 'reports/segmentation-experiment.json').read_text())
        return (review, experiment)

    def summary(self):
        review, experiment = self.evidence()
        predictions = {r['record_id']: r for r in experiment['rows']}
        return {'review_version': review['proposal_sha256'], 'summary': review['summary'], 'experiment': {k: experiment[k] for k in ('summary', 'comparison_to_Change', 'mask_quality')}, 'rows': [{**r, 'group': predictions[r['record_id']]['group'], 'predictions': predictions[r['record_id']]['predictions']} for r in review['rows']]}

    @lru_cache(maxsize=1)
    def image_paths(self):
        paths = list((self.research / 'lefood').rglob('*.JPG'))
        indexed = {p.name: p for p in paths}
        if len(indexed) != len(paths):
            raise ValueError('Ambiguous source image basenames.')
        return indexed

    @lru_cache(maxsize=64)
    def image(self, record_id, kind, overlay=False):
        if not re.fullmatch('L[0-9]+', record_id) or kind not in ('before', 'after'):
            raise ValueError('Unknown review image.')
        review, _ = self.evidence()
        row = next((r for r in review['rows'] if r['record_id'] == record_id), None)
        if row is None:
            raise ValueError('Unknown review record.')
        item = row['images'][kind]
        path = self.image_paths().get(item['filename'])
        if path is None:
            raise FileNotFoundError('Original LeFood photos are not installed in the local research folder.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Source image differs from reviewed evidence.')
        with Image.open(path) as im:
            photo = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (1000, 1000))
        if overlay:
            mask_path = self.root / f'data/segmentation-cache/mobilenet-v1/masks/{record_id}_{kind}.png'
            with Image.open(mask_path) as im:
                mask = im.convert('L').resize(photo.size, Image.Resampling.NEAREST).point(lambda v: int(v * 0.35))
            photo = Image.composite(Image.new('RGB', photo.size, (0, 140, 255)), photo, mask)
        output = io.BytesIO()
        photo.save(output, format='JPEG', quality=90)
        return output.getvalue()
