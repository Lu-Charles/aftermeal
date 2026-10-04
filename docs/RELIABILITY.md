# Input validation

The hosted upload API accepts before/after JPEG or PNG bytes at `POST /api/upload`. Request formats and HTTP errors are listed in [DEPLOY.md](DEPLOY.md#api).

## Image checks

The server applies EXIF orientation and converts images to RGB. It rejects unreadable images, constant-color photos, identical decoded pairs and images with either side below 224 pixels. Hosted images are limited to 4 MB and 4 megapixels each; the browser resizes large originals before sending them.

Duplicate detection hashes oriented RGB pixels and dimensions. It catches metadata-only changes but can miss re-encoded JPEGs and near duplicates.

`data/input_review_flags.json` contains one flagged starting image, L476. Its source photo appears empty despite a positive recorded weight. An exact match returns `needs_review`; the source record remains in the benchmark.

## Responses

| Status | Meaning | Estimate |
|---|---|---|
| `estimated` | Validation and inference completed | Returned |
| `needs_review` | Starting image matches a flagged source photo | Withheld |
| `rejected` | Duplicate, constant-color or undersized images | Withheld |

These outcomes return HTTP 200. Read `status` before accessing `estimated_fraction`. The `input_checks` field includes issue codes and decoded-image hashes.

Checks run before photo encoding. Changing either photo clears the previous result, and the browser discards responses from older requests.

## Coverage

Tests cover metadata-only duplicate changes, exact-image flags, invalid requests, image limits and concurrent uploads. A [scan of the 524 LeFood pairs](../reports/input-checks.json) found no duplicate, constant-color or undersized pairs; L476 matched its registry entry.

These checks validate image structure, not whether the photos show the same serving. The interface asks users to photograph the same meal with similar framing.
