# Development

For the running app, visit [aftermeal.onrender.com](https://aftermeal.onrender.com).

## Setup and tests

Use Python 3.12. From the repository root on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-public.txt
python -m foodvision verify
python -m unittest discover -s tests -v
```

The tests and cached-feature benchmark run without PyTorch or a GPU.

## Run the website

The Docker image provides the same photo-upload service as Render:

```bash
docker build -t aftermeal .
docker run --rm -p 127.0.0.1:8080:8080 --memory=512m --memory-swap=512m --read-only --tmpfs /tmp aftermeal
```

Open http://127.0.0.1:8080. The first build downloads the encoder and checks the exported model against PyTorch. [Container settings and API](DEPLOY.md).

For a faster UI preview using sample pairs only:

```bash
python -m gunicorn --config gunicorn.conf.py --bind 127.0.0.1:8080 'foodvision.public:create_app()'
```

Leave `AFTERMEAL_HOSTED_UPLOADS` unset for this preview. Use Docker to test uploads.

## Reproduce the benchmark

```bash
python -m foodvision benchmark --output .local/benchmark.json
python -m foodvision diagnose --output .local/diagnostics.json
```

The benchmark refits the regressors and checks all ten outer episodes against saved forecasts within `1e-10`. Diagnostics report errors for zero-mass and small-leftover cases. Choose a new output filename when rerunning; these commands refuse to overwrite existing results.

Segmentation and depth experiments require additional source photos, weights and caches. Setup is documented with the [segmentation](EXPERIMENTS.md#reproduction) and [depth](DEPTH.md#reproduce) results.

## Release checks

```bash
node --check --input-type=module < web/app.js
python -m gunicorn --check-config --config gunicorn.conf.py 'foodvision.public:create_app()'
python scripts/smoke_public.py http://127.0.0.1:8080
```

Run the smoke test against the Docker container. It checks all four samples, request limits, blocked routes and an actual uploaded photo pair. CI also builds and tests the container with 512 MB of memory and 0.1 CPU.
