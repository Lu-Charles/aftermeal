# Hosting Aftermeal

The website runs at https://aftermeal.onrender.com. Render builds the connected repository's Dockerfile from `main`. The image includes the pinned DINOv2 encoder converted to ONNX and the existing regression head. Inference runs on CPU and uses no persistent storage.

## Container and Render

```bash
docker build -t aftermeal:1.1 .
docker run --rm -p 127.0.0.1:8080:8080 --memory=512m --memory-swap=512m --read-only --tmpfs /tmp aftermeal:1.1
python scripts/smoke_public.py http://127.0.0.1:8080
```

Use Render's Docker web service, `main` branch, and `/healthz` health check. The image listens on `PORT` (8080 by default), runs as UID 10001, and enables `AFTERMEAL_HOSTED_UPLOADS=1`. Free instances sleep when idle, so the first visit can take about a minute to wake. The build exports the encoder, verifies its checksum and compares predictions with PyTorch on all four sample pairs. DINOv2's license is copied alongside the exported model.

Gunicorn runs one model process with four request threads. Only one upload can read its body or run inference at a time; other threads serve health and sample requests. The process accepts up to 12 upload attempts per minute. Busy requests return 429 with `Retry-After`. Increasing workers duplicates model memory. Keep one worker on the 512 MB instance. Render provides HTTPS.

The CI container smoke runs with a 512 MB memory limit and 0.1 CPU, verifies actual photo inference, and checks that PyTorch is absent from the serving image. The browser permits 180 seconds for upload analysis and 90 seconds for initial configuration after a cold start. Aborting a browser request does not cancel work already running on the server.

## Upload behavior

Choose **Your photos**, select before/after JPG or PNG images, then **Analyze pair**. The browser accepts originals up to 8 MB and scales large images to at most 1600 pixels on their longest side before upload. The server independently limits each image to 4 MB and 4 megapixels, applies EXIF orientation, and checks duplicate, invalid, constant, undersized and known flagged images. Encoding uses a 256-pixel resize followed by a 224-pixel center crop. Resizing a large original before upload can slightly change its prediction.

The photo pair is sent over HTTPS and processed in memory without saving photos or results. Access logs omit request bodies, client IPs, queries, referrers and user agents. Render may have its own infrastructure logs. Food highlights are available for sample pairs only.

## API

| Route | Contract |
|---|---|
| `GET /healthz` | Version, mode, upload capability; model loaded and warmed in hosted mode |
| `GET /api/config` | Available capabilities and sample/model hashes |
| `GET /api/examples` | Four available photo pairs |
| `POST /api/predict` | JSON with exactly one string `example_id`; maximum 1 KB |
| `POST /api/upload` | JSON with exactly `before` and `after`, each plain base64 image bytes; maximum 12 MB request body |
| `GET /api/benchmark` | Saved benchmark result |
| `GET /case-study` | Compatibility entry into the main app's About view |

Uploads return the estimated fraction and input checks, or a rejected/review status without an estimate. Invalid input returns 400, cross-host browser origins 403, oversized bodies 413, wrong content type 415, busy/rate-limited service 429, and unavailable inference 503. `/lab`, `/capture`, `/api/highlight` and collection APIs are absent from the hosted service.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for environment setup, a sample-only preview and tests.

## Source package

```bash
python scripts/package_release.py .local/releases/aftermeal-1.1.0.zip
```

The allowlisted source ZIP excludes environments, downloaded/exported weights, caches, local databases and personal records. The Docker build regenerates the encoder. `RELEASE-MANIFEST.json` hashes packaged files and records the dataset-relative workbook path in the packaged source audit. Data attribution is in [DATA.md](DATA.md).
