# Visual review provenance

Review lab exports use schema 2. Each record carries before/after observations,
image hashes, evidence version, reviewer description and timezone-aware timestamps.
Exposure to source weights, overlays and AI proposals is recorded separately.
Visual observations never certify corrected mass or evaluation gold.

Import with `python -m foodvision import-reviews --input reviews.json`.
