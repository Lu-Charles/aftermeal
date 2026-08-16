# Input checks and review history

## Upload contract

`POST /api/predict` accepts JSON with base64 `before` and `after` image bytes and optional `starting_portion`: `visible_food`, `empty_or_residue`, or `uncertain`. This last value records the user's observation, not model detection. The upload UI sends no observation and analyzes valid photo pairs directly. Missing or uncertain observations do not block inference and are never converted to a positive food-presence assertion.

| Response status | Meaning | Percentage returned? |
|---|---|---|
| `estimated` | Upload passed policy checks and encoding completed | Yes |
| `needs_review` | Starting portion is explicitly described as empty through the API, or matches an unresolved source flag | No |
| `rejected` | Identical decoded photos, constant-color image, or either side below 224 pixels | No |

These outcomes return HTTP 200 with `input_checks` containing policy version, issues, decoded-image hashes and the observation's source. Malformed requests return HTTP 400; unavailable encoding dependencies return HTTP 503. Clients must branch on `status` before reading `estimated_fraction`. Historical `example_id` requests retain their existing response format and cached predictions.

Checks run before loading the encoder. The UI removes previous results when a photo changes and discards stale in-flight responses. Issue links focus the relevant input for recovery.

The minimum resolution is an input policy based on the encoder's 224-pixel crop, not a measured accuracy threshold. Duplicate detection hashes oriented RGB pixels plus dimensions, so changing PNG metadata cannot bypass it. It does not detect near duplicates, re-encoded JPEGs or mismatched servings. Constant-color rejection is not an empty-plate classifier.

`data/input_review_flags.json` currently contains one exact-image flag, L476, with raw/decoded image hashes and its AI-review provenance. This flag is unverified and conservative; it does not establish that every other starting photo contains food. Reframed or re-encoded copies may not match. The original benchmark and model assets remain unchanged.

## Import visual reviews

In Review lab, select an observation for both photos, save, and export. Download the JSON or save the copyable JSON as a UTF-8 file. Then:

```bash
python -m foodvision import-reviews --input reviews.json
python -m foodvision review-status
```

Both commands accept `--db path/to/reviews.sqlite3`. The default `.local/` directory is ignored by Git. Back up the exported JSON or database to preserve work beyond this computer. Importing does not synchronize annotations back into browser storage.

The schema-2 importer validates the entire batch against the review queue: record IDs, both raw image hashes, evidence version, observation values, booleans, timestamps and note limits. Older exports lacking a before observation are rejected. `mass_verified` and `evaluation_gold` must both be false. Reviewer descriptions and exposure flags are self-reported; this local workflow provides no identity authentication.

SQLite stores the canonical export, its SHA-256 digest, individual review revisions, and export-to-review links. One transaction commits each batch. A repeated export returns `already_imported`; re-exporting unchanged reviews creates no new revisions. Changed reviews append history. Concurrent imports are serialized, and any storage failure rolls back the batch. Status chooses the latest observation timestamp per record/evidence version, so a later import of an older backup cannot hide a newer observation; import order breaks timestamp ties. This is an application-level history, not a tamper-proof database or multi-reviewer consensus system.

## Verification

The v2 upload flow removes the manual question. Current tests confirm that missing/uncertain observations pass structural checks with their original provenance intact, and an API request without an observation reaches the encoder. The following audit describes the earlier v1 workflow.

- Automated cases cover duplicate pixels with different metadata, missing/uncertain starting portions, exact-image flags and direct API requests. Rejected/review cases assert that encoding is never invoked.
- The [524-pair structural audit](../reports/input-checks.json) found no duplicate/constant/undersized pairs; L476 matched the known review flag. This audit supplies `visible_food` to isolate structural checks, so it does **not** measure empty-plate detection performance.
- Real CPU inference on uploaded L492 images returned approximately 29.1% after the user-observation gate. Browser checks covered rejection, recovery, clearing stale results, and mobile layout. This is a functional smoke test, not new accuracy evidence.
- Import tests cover concurrent retries, revisions arriving out of order, corrupted evidence, invalid gold claims, and rollback after an injected write failure. All annotation fixtures are explicitly synthetic and live only in temporary test databases. No actual human-reviewed label set was created by these tests.

The next accuracy step is independent review of starting photos and residue cases, followed by a grouped evaluation with defensible labels. AI proposals can prioritize that review; they cannot validate themselves.
