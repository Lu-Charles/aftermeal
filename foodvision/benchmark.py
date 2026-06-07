"""Recompute a compact grouped benchmark from the bundled DINOv2 features."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent.parent
METHODS = ('Mean', 'Median', 'Appearance', 'Change')

def verify_payload(root: Path=ROOT) -> int:
    manifest = json.loads((root / 'data/checksums.json').read_text())
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Bundled input is missing or changed: {name}')
    return len(manifest)
