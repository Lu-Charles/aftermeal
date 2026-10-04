"""Memory-bounded photo inference using the same DINOv2 weights and regressor."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .inference import Predictor, ROOT, decode_image
from .input_checks import inspect_pair, load_flags

MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_UPLOAD_PIXELS = 4_000_000


def preprocess(image):
    """Match torchvision's PIL Resize(256), CenterCrop(224), and normalization."""
    width, height = image.size
    if width <= height:
        size = (256, int(256 * height / width))
    else:
        size = (int(256 * width / height), 256)
    resized = image.resize(size, Image.Resampling.BICUBIC)
    left, top = round((size[0] - 224) / 2), round((size[1] - 224) / 2)
    pixels = np.asarray(resized.crop((left, top, left + 224, top + 224)), dtype=np.float32)
    pixels = pixels.transpose(2, 0, 1) / np.float32(255)
    mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(((pixels - mean) / std)[None])


class HostedPredictor(Predictor):
    def __init__(self, root=ROOT, encoder_directory=None):
        super().__init__(root)
        import onnxruntime as ort
        directory = Path(encoder_directory) if encoder_directory else root / 'models/hosted'
        metadata = json.loads((directory / 'encoder.json').read_text())
        path = directory / 'encoder.onnx'
        # Stream the hash so startup does not allocate a second full model copy.
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != metadata['sha256']:
            raise RuntimeError('Hosted encoder checksum failed.')
        for field in ('encoder_revision', 'encoder_weights_sha256'):
            if metadata[field] != self.metadata[field]:
                raise RuntimeError('Hosted encoder does not match the calibrated model.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.flags = load_flags(root)
        self.session.run(['embedding'], {'photo': np.zeros((1, 3, 224, 224), dtype=np.float32)})

    def photos(self, before, after, starting_portion=None):
        if any(not raw or len(raw) > MAX_UPLOAD_BYTES for raw in (before, after)):
            raise ValueError('Each uploaded photo must be nonempty and at most 4 MB.')
        images = [decode_image(raw, max_pixels=MAX_UPLOAD_PIXELS) for raw in (before, after)]
        checks = inspect_pair(images, starting_portion, self.flags)
        if checks['status'] != 'ready':
            return {'status': checks['status'], 'input_checks': checks,
                    'input_mode': 'hosted photo validation'}
        # Encode sequentially: batching two images raises the peak activation memory.
        embeddings = [self.session.run(['embedding'], {'photo': preprocess(image)})[0]
                      for image in images]
        fraction = float(self.predict_embeddings(*embeddings)[0])
        return {'status': 'estimated', 'estimated_fraction': fraction,
                'input_mode': 'hosted photo encoding', 'input_checks': checks,
                'photos_stored': False}
