# Weighed-data pilot

Open `/capture` from **Collect data** in the workspace. Start with ten independent servings, each with a starting photo/weight and at least one after photo/weight. This is a collection-process pilot, not enough data to certify accuracy. Additional stages of one serving are useful but do not count as independent servings.

## Collection procedure

1. Create a named session for one capture setup/occasion. Choose **Pilot / development** for the first ten servings. Select **Reserved evaluation** before collecting future evaluation sessions; their purpose cannot be changed in this interface.
2. Use a scale on a stable surface. Record its smallest increment. Weigh the empty, dry plate without cutlery.
3. Add a serving and record the total plate-plus-food display. Photograph the whole plate from a consistent angle and distance. Save this starting record.
4. After eating or removing some food, record the new total display and take a new photo. Keep the same plate and avoid adding food or changing cutlery. Do not subtract the plate yourself: the app calculates net food mass and the after/before fraction.
5. Include substantial portions, tiny leftovers and thin residue. Record visible material separately from weight. A residue observation never overwrites a scale reading with zero. If the scale reads the plate's tare, the net reading is zero, but food below the scale's resolution may still remain.
6. Export a ZIP backup. Finish the pilot by checking capture consistency and transcription errors before defining a larger training and evaluation collection.

After readings can be corrected with a required reason. The previous values remain in their revision history. An incorrect starting weight, tare or photo requires excluding the serving and creating a new one. Exclusion retains the original evidence and reason. Images cannot be silently replaced through a correction.

## Storage and boundaries

The app explicitly persists collection data to `.local/capture.sqlite3`, ignored by Git. SQLite stores original image bytes and their raw/decoded-pixel hashes, sessions, servings, readings, revisions and request deduplication records. Collection photos may retain their original image metadata. Workspace analysis uploads retain their existing in-memory behavior; they are not automatically copied into the collection.

Each mutation is atomic, including its image. Repeating an identical request ID returns the original result; reusing the ID with a changed payload fails. A correction must name the revision it replaces, preventing one open tab from silently overwriting another. Duplicate decoded photographs are rejected across all servings and sessions. This is exact-image protection, not perceptual near-duplicate detection.

Readings use integer milligrams internally, with decimal gram input validated against the declared scale increment. Before net mass must be positive; after mass cannot be negative or exceed the starting food mass. Collection does not infer model predictions, invoke an LLM, alter source datasets or automatically train a model.

Pilot and evaluation purposes are immutable through the API. All stages share their serving/session IDs. Any future training importer must exclude reserved evaluation sessions and excluded servings, and group related records by session/serving. This collection UI records that boundary; it is not an independently audited holdout or an implemented training pipeline.

## Portable export

`GET /api/capture/export` creates `aftermeal-capture.zip` from a consistent database snapshot. It includes:

- `manifest.json`: schema version, user-recorded measurement status, session purposes, serving IDs, current net masses/fractions, all reading revisions and photo hashes.
- `photos/`: original JPEG/PNG image bytes, with paths matching the manifest.
- `README.txt`: units, exclusions and grouping rules.

Treat the latest reading revision as current, retain prior revisions for audit, and filter exclusions explicitly. The archive is a portable evidence package; automatic restore/import is not implemented. Back up the SQLite database while the server is stopped if you need a complete local-app restore.

The server binds to loopback. Capture routes reject external origins and non-loopback Host values. There is no multi-user authentication or public deployment in this feature.

## Verification

Twelve capture tests cover tare subtraction, decimal precision, same-image contamination, rollback, retries, stale corrections, exclusions, ZIP hashes and HTTP origin/Host checks. The complete suite passes 79 tests. A browser smoke test used a separate temporary database, marked simulated readings, and checked before/after creation, correction history across reload, ZIP download and a 390-pixel mobile layout. No simulated servings were inserted into the user's collection.
