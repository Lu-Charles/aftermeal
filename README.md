# Aftermeal — Food Leftover Estimation

Estimate the fraction of food remaining from before-and-after meal photos.
The target is the recorded mass after eating divided by the starting food mass.
Photos alone do not establish grams or environmental savings.

## Setup

Use Python 3.12 and install `requirements.txt` in a virtual environment.

## Data

Sample photographs are from LeFood-Set v1 (CC BY 4.0). Derived ACETADA
features retain CC BY-NC 4.0 terms. Code and data licenses are separate.
See [data attribution](docs/DATA.md) and [license](LICENSE).

## Benchmark

Run `python -m foodvision verify` and
`python -m foodvision benchmark --output reports/my-benchmark.json`.
Evaluation separates food/participant groups and similar-image components.
Candidate selection uses inner validation only. Existing outputs are not overwritten.

Five folds per collection use 50 calibration labels per fit. Change-feature
group-balanced MAE is 9.6330 percentage points on LeFood and 9.4585 on ACETADA.
Appearance features perform slightly better on ACETADA. These results measure
within-collection calibration, not transfer to new settings.

See [model details](docs/MODEL.md) and [benchmark results](reports/benchmark.json).

## Local demo

Run `python -m foodvision demo` and open http://127.0.0.1:8765.
Four held-out samples run from saved embeddings without PyTorch or model downloads.

For new photos, install `requirements-images.txt`. Use
`python -m foodvision predict --before before.jpg --after after.jpg`.
The optional frozen DINOv2 encoder runs locally and downloads pinned weights separately.

The browser also accepts JPEG and PNG photo pairs for local analysis.

Optional food highlights are an inspection aid; they do not change percentage estimates.
