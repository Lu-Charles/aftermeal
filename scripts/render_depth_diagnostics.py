"""Render the four existing gallery cases, without selecting examples by outcome."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def render(cache, report_path, output):
    root = Path(__file__).resolve().parents[1]
    report = json.loads(report_path.read_text())
    rows = {r['record_id']: r for r in report['rows']}
    cases = ['L81', 'L133', 'L388', 'L492']
    fig, axes = plt.subplots(4, 4, figsize=(12, 10.8))
    for i, record_id in enumerate(cases):
        row = rows[record_id]
        for j, role in enumerate(('before', 'after')):
            with Image.open(root / 'examples/images' / f'{record_id}_{role}.jpg') as photo:
                axes[i, j].imshow(photo)
            with np.load(cache / 'maps' / f'{record_id}_{role}.npz', allow_pickle=False) as data:
                depth = data['inverse_depth']
            low, high = np.quantile(depth, [0.05, 0.95])
            axes[i, j + 2].imshow(depth, cmap='magma', vmin=low, vmax=high)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[i, 0].set_ylabel(record_id, fontsize=12, weight='bold')
        axes[i, 0].set_xlabel(f"Recorded: {100 * row['fraction']:.1f}%", fontsize=10)
        axes[i, 1].set_xlabel(f"Current model: {100 * row['predictions']['Change']:.1f}%", fontsize=10)
        axes[i, 2].set_xlabel(f"Change + depth: {100 * row['predictions']['ChangeDepth']:.1f}%", fontsize=10)
        axes[i, 3].set_xlabel(f"Change + mask + depth: {100 * row['predictions']['ChangeMaskDepth']:.1f}%", fontsize=10)
    for ax, title in zip(axes[0], ('Before photo', 'After photo', 'Before relative depth', 'After relative depth')):
        ax.set_title(title, fontsize=12, pad=12)
    fig.suptitle('Depth experiment · existing gallery cases', fontsize=18, y=0.985)
    fig.text(0.5, 0.015, 'Brighter = predicted nearer. Each map has its own scale; brightness is not comparable across photos.\nPredicted depth is not measured height or volume. These development examples are not an accuracy benchmark.', ha='center', fontsize=10, color='#444444')
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(output, dpi=150, facecolor='white', metadata={'Software': None})
    plt.close(fig)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.cache, args.report, args.output)
