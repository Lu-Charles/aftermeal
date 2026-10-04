"""Export the pinned DINOv2 encoder for a small CPU-only serving container."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from foodvision.inference import image_encoder, Predictor
from foodvision.hosted import preprocess


class Embedding(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, photo):
        return self.encoder.forward_features(photo, masks=None)['x_norm_clstoken']


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    encoder, transform, _ = image_encoder()
    path = output / 'encoder.onnx'
    # The app always feeds one 224-pixel crop; fixed shapes bound serving memory.
    with torch.inference_mode():
        torch.onnx.export(Embedding(encoder).eval(), torch.zeros(1, 3, 224, 224), str(path),
                          input_names=['photo'], output_names=['embedding'],
                          opset_version=17, dynamo=False, external_data=False)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    predictor = Predictor()
    parity = []
    for row in json.loads((ROOT / 'examples/examples.json').read_text()):
        actual, expected = [], []
        for role in ('before', 'after'):
            with Image.open(ROOT / row[role].lstrip('/')) as photo:
                batch = transform(photo.convert('RGB'))[None]
                np.testing.assert_array_equal(preprocess(photo.convert('RGB')), batch.numpy())
            with torch.inference_mode():
                reference = encoder(batch).cpu().numpy()
            result = session.run(['embedding'], {'photo': batch.numpy()})[0]
            np.testing.assert_allclose(result, reference, rtol=5e-4, atol=2e-4)
            actual.append(result)
            expected.append(reference)
        result = float(predictor.predict_embeddings(*actual)[0])
        reference = float(predictor.predict_embeddings(*expected)[0])
        difference = abs(result - reference)
        if difference > 1e-4:
            raise ValueError('Export changed a sample prediction beyond tolerance.')
        parity.append({'id': row['id'], 'onnx_fraction': result,
                       'pytorch_fraction': reference, 'absolute_difference': difference})
    metadata = predictor.metadata
    license_path = Path(torch.hub.get_dir()) / ('facebookresearch_dinov2_' + metadata['encoder_revision']) / 'LICENSE'
    (output / 'LICENSE.dinov2').write_bytes(license_path.read_bytes())
    manifest = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'encoder_revision': metadata['encoder_revision'],
                'encoder_weights_sha256': metadata['encoder_weights_sha256'],
                'input_shape': [1, 3, 224, 224], 'output_shape': [1, 384],
                'sample_parity': parity}
    (output / 'encoder.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/hosted')
    export(parser.parse_args().output)
