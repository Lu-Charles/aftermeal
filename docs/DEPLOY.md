# Deploying Aftermeal 1.0

The public release is a sample analysis demo. Four held-out pairs use the same compact regression head as local inference; predictions are computed from saved embeddings at worker startup. Upload analysis, original research-image access and SQLite collection remain local tools. No GPU, external API, RunPod account or persistent disk is required for the public service.

## Run the public service

From this project directory (or the extracted standalone release), use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-public.txt
python -m gunicorn --config gunicorn.conf.py --bind 127.0.0.1:8080 'foodvision.public:create_app()'
```

Open http://127.0.0.1:8080. In another terminal:

```bash
python scripts/smoke_public.py http://127.0.0.1:8080
```

For a container host:

```bash
docker build -t aftermeal:1.0 .
docker run --rm -p 127.0.0.1:8080:8080 --read-only --tmpfs /tmp aftermeal:1.0
```

The Docker image listens on `PORT` (8080 by default) on all container interfaces, runs as UID 10001, and uses `/healthz` as its health check. A hosting service should provide HTTPS and reverse-proxy request/connection limits. Set a 2 KB proxy body limit; the app enforces a stricter 1 KB JSON limit. Do not expose the local `python -m foodvision demo` server publicly.

Gunicorn has two synchronous workers, a backlog of 64 and a 15-second timeout. Requests serve preloaded bytes or select one of four predictions: there is no heavy inference queue. Slow or stalled workers are replaced. These settings bound work per process, but do not establish production capacity or provide distributed rate limiting. Host-level traffic controls are required before a broad launch. [Gunicorn settings reference](https://gunicorn.org/reference/settings/).

## API

| Route | Contract |
|---|---|
| `GET /healthz` | Worker booted successfully, assets verified, version and mode |
| `GET /api/config` | Available capabilities and sample/model hashes |
| `GET /api/examples` | Four available photo pairs |
| `POST /api/predict` | JSON with exactly one string `example_id`; maximum 1 KB |
| `GET /api/benchmark` | Saved benchmark result; reproduction is a separate CLI command |
| `GET /case-study` | Compatibility entry into the main app’s About view |

`HEAD` is supported for GET resources. Invalid JSON/IDs return 400, wrong content type 415, excess body length 413, cross-host browser origins 403, unsupported prediction methods 405, and unavailable/private routes 404. Uploads, `/lab`, `/capture`, and their APIs are absent from the public application, even if requested directly. The public app never initializes a capture database.

The UI times out sample requests after 10 seconds and optional local upload/highlight requests after 120 seconds. Aborting a local browser request does not interrupt an already-running encoder download or computation. A slow highlight no longer blocks display of a completed estimate.

## Data and privacy

The public service has no analytics, upload, login or review-submission endpoint and does not store visitor input. App access logs omit client IPs, query strings, referrers, user agents and request bodies. The hosting provider may have separate logs. See [data attribution](DATA.md) for asset licenses. Local research-lab reviews and weighed collection are separate tools and are not served publicly.

## Repository and source package

This working directory sits inside a larger local workspace. Publish the **standalone `aftermeal/` package**, not its parent directory. The workflow at `.github/workflows/ci.yml` expects this project to be the repository root.

```bash
python scripts/package_release.py .local/releases/aftermeal-1.0.2.zip
```

The packaging script uses an allowlist and refuses to replace an existing ZIP. It excludes environments, downloaded weights, feature/mask caches, `.local` databases, exported personal records and unrelated projects. It includes existing numerical experiment reports and attributed LeFood figures. The original workbook's absolute local path is redacted in the packaged source-audit report; `RELEASE-MANIFEST.json` records that change and hashes every packaged file. The source workspace and original reports stay unchanged.

## Validation and remaining publication steps

Release verification is recorded in [RELEASE.md](RELEASE.md). CI installs only public dependencies, runs Python tests and JavaScript syntax checks, reproduces the full feature benchmark, validates Gunicorn configuration, builds the container and runs its black-box smoke test.

Before publication, choose the repository and hosting account, inspect the packaged files, and configure the host's HTTPS and traffic limits. No repository push, hosted service, paid resource or public URL has been created by preparing this release. The Docker daemon was unavailable during local verification; container build/run remains to be verified in CI or on a Docker-enabled host. The same Gunicorn application was exercised locally from a fresh environment.
