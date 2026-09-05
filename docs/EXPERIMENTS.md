# Food-mask experiment and AI review

Experiments on frozen image, segmentation and depth features.

## Outcome

The first frozen segmentation model did not improve remaining-mass estimation. Keep the existing demo model. This is evidence against this particular checkpoint, preprocessing and feature combination, not evidence that all segmentation approaches fail.

| Method | All 524 pairs: group MAE (pp) | Recorded-zero MAE (pp) | Small-positive MAE (pp) | Recorded zeros predicted >10% |
|---|---:|---:|---:|---:|
| Image change | 9.63 | 4.01 | 11.12 | 10 / 207 |
| Appearance | 14.65 | 6.13 | 12.36 | 47 / 207 |
| Segmentation | 15.82 | 14.47 | 15.88 | 108 / 207 |
| Combined | 11.94 | 9.78 | 15.56 | 78 / 207 |

Small-positive means recorded remaining fraction >0 and <=10% (32 pairs). The segmentation and combined methods fail all four provisional adoption gates. Their MAE differences against image change are +6.19 pp (conditional 95% interval -0.94 to +8.74) and +2.30 pp (-2.06 to +3.91). Intervals use 2,000 paired resamples of 16 dependency blocks, retaining original group weighting; they do not establish transfer or account for label errors. One block contains 18 of the 34 food groups, limiting precision.

All 524 source labels remain unchanged. The original five grouped folds and 50-label budgets are retained, with train-only scaling and inner-only candidate selection. Original baseline forecasts match archived references within 1e-10. No outer labels choose features or hyperparameters. This is retrospective development, not an untouched final evaluation.

The [frozen protocol](SEGMENTATION-PROTOCOL.json) was saved before extraction/evaluation. A [documented amendment](SEGMENTATION-AMENDMENT.json) corrects an overly strict dependency-block check: the original protocol purges record-level similarity components at fitting boundaries and uses broader group-union blocks for uncertainty. The first run stopped before scoring; original folds and all feature/model choices stayed fixed. Raw image SHA256, record, food-group and similarity-component separation are checked at every inner and outer boundary. Serving identity beyond those links is not established.

Full forecasts, selections and metrics: [experiment report](../reports/segmentation-experiment.json).

## Mask inspection

The frozen [DeepLabV3Plus/MobileNetV2 checkpoint](https://huggingface.co/mawiie/food-segmentation-mobilenet) has 104 classes; any non-background argmax is treated as predicted food. In the six preselected diagnostic cases, it often includes rice residue and stained plate areas. This helps explain why area features are unsuitable without better mask quality. The overlays are unchanged model outputs, not traced illustrations or explanations of DINOv2.

![Diagnostic predicted food masks](../reports/segmentation-preview.jpg)

There are no independently verified masks here, so mask IoU and confidence are not established. The author model card declares MIT for the checkpoint; FoodSeg103 is its stated training source. The model card's license does not resolve separate dataset terms. No FoodSeg103 data were downloaded or redistributed. Weights are local, ignored, and not bundled in a release.

## AI-assisted review

All 524 pairs were screened with automated visual review on 22 contact sheets with IDs and photos, without predictions or weights displayed. Some pairs were already familiar from previous diagnostics; this is not independent blinded adjudication. Labels are broad visual proposals: visible food-like material (285), no clear portion/residue/crumbs (205), or uncertain (34). No mass guesses were generated.

Joining those proposals to unchanged source measurements creates **48 review priorities**, including 34 ambiguous-material cases and 14 possible source conflicts. Seven pairs contain visible material despite zero recorded after mass; six show no clear portion despite >10% recorded remaining mass; one starting photo looks empty despite positive mass. These are unverified flags, not 14 proven dataset errors. Bone, inedible material, scale resolution, residue definitions and pairing errors need investigation.

- [All proposals and source provenance](../reports/review-queue/review-queue.json)
- [Source-conflict proposals, page 1](../reports/review-queue/source-conflict-proposals-01.jpg), [page 2](../reports/review-queue/source-conflict-proposals-02.jpg)
- [Original AI review codes and rubric](../reports/ai-review/page-proposals.json)

Open **Review lab** from the app's Benchmark view, or visit `/lab`. Search by record/food, filter cases, compare original photos, optionally show predicted masks, and save observations for **both** photos locally in the browser. Clean-empty, residue-only or uncertain before images set `starting_portion_unverified`; they do not establish that a meal was consumed. Source weights and predictions start collapsed. Export JSON records include image hashes, evidence version, timestamp and whether source labels, AI proposals or masks were viewed. The export panel offers a file download and copyable JSON for browsers that do not support it. Browser reviews are not verified mass labels or an independently adjudicated gold set. Clearing browser storage removes local reviews; export to retain a copy.

The lab reads original photos from the sibling `photo-waste-feasibility/lefood` folder and predicted masks from the ignored cache. Other machines can read the saved numerical reports without those images; the lab reports missing local assets. It serves only allowlisted record/role images, checks their original hashes, and exposes no arbitrary filesystem route.

Photos: LeFood-Set v1, Yuita Arum Sari, Yudi Arimba Wani and Atsushi Nakazawa, CC BY 4.0. Contact sheets are resized; previews add predicted blue overlays.

## Reproduction

The demo's dependencies are unchanged. Use a separate optional environment for the segmenter:

```bash
python3.12 -m venv .venv-segmentation
.venv-segmentation/bin/python -m pip install -r requirements-segmentation.txt
```

Download `best_model.pth` from revision `93d8c84b121ecce76ed752be538e01d72df15825` of `mawiie/food-segmentation-mobilenet` into `models/segmentation-cache/`. Expected SHA256: `6ecc4da2210d8a2128f6a79670d40f25c8f5ffbf11b3358689669c431896600f`. Loading verifies the checksum and uses restricted `weights_only=True` state-dict loading. The 512-square full-frame resize differs from the author's validation center crop; this transfer choice was fixed before evaluation. No threshold/postprocessing was fitted.

```bash
.venv-segmentation/bin/python -m foodvision segment-cache \
  --research-root ../photo-waste-feasibility --output data/segmentation-cache/my-run
python -m foodvision segmentation-experiment \
  --cache data/segmentation-cache/my-run --output reports/my-segmentation.json
python scripts/build_review_queue.py \
  --research-root ../photo-waste-feasibility \
  --proposals reports/ai-review/page-proposals.json \
  --manifest data/segmentation-cache/my-run/manifest.json --output reports/my-review-queue
```

Extraction processes 1,048 images, stores 19 paired mask features, original image provenance, model revision/checksum, environment versions and feature/protocol checksums. On this CPU run it took about 554 seconds; this is extraction time on this machine, not an upload latency guarantee.

## Next accuracy milestone

Resolve the 14 source-conflict proposals and independently annotate food/residue masks before selecting a replacement segmenter. Keep every historical mass result intact. A different pretrained checkpoint alone is not a validated improvement. For claims about new meals, collect a separate weighed serving/session holdout with clean plates, residue, tiny leftovers, piles and mixed dishes. The app's review tool supports this evidence-building process; it does not finish it automatically.

## Main demo overlays

The four sample pairs now include the frozen experimental masks in the main workspace, with a switch to show original photographs. `scripts/export_sample_overlays.py` renders eight bundled overlays from the existing cache and records image/model provenance in `examples/overlay_manifest.json`. Masks are not retouched and do not alter percentage predictions. No additional segmentation accuracy is claimed. The subsequent live-upload integration also runs this frozen segmenter on uploaded image bytes. The mask remains separate from the percentage estimator.

## Expanded training and zero-mass modeling

Two new development experiments were specified before their runs, retaining the original outer test IDs and source targets. These datasets have already informed development; this is not an untouched final evaluation.

| Model | LeFood error (points) | ACETADA error (points) |
|---|---:|---:|
| Original Change, 50 calibration pairs | 9.63 | 9.46 |
| Change, expanded training pool | 9.75 | 8.39 |
| Appearance, expanded training pool | 11.07 | 7.82 |
| Two-stage mass model, 50 pairs | 9.46 | 9.03 |
| Two-stage mass model, expanded pool | 11.46 | 7.77 |

Expanded training uses 353–462 LeFood pairs and 621–650 ACETADA pairs per fold. Outer and inner fitting boundaries exclude shared groups/similarity components; LeFood additionally excludes exact raw image hashes across before/after roles. Inner partitions are constructed without using target values. The two-stage model learns a positive-mass score from after-image embeddings and a conditional amount model from positive-mass training records. Their product predicts remaining fraction; there is no manually chosen zero cutoff, and the score is not a calibrated food-presence confidence.

No tested LeFood candidate passed all predeclared adoption gates. In particular, the 50-pair two-stage result improves error by only 0.18 points, with a paired block interval crossing zero. For L81, it predicts 64.7% versus 26.6% recorded, compared with the original 80.1%. Improving that one case is not a reason to promote it. The original estimator remains deployed.

Reproduce with:

```bash
python -m foodvision.data_budget reports/my-data-budget.json
python -m foodvision.hurdle reports/my-hurdle.json
```

The hurdle run uses the checked-in `reports/data-budget.json` for its expanded partitions and records that report’s hash. Protocols are `docs/DATA-BUDGET-PROTOCOL.json` and `docs/HURDLE-PROTOCOL.json`; full forecasts, selections, slices and intervals are in the matching reports. Tests check target independence, group/image purging and the no-positive-label edge case.

## Live upload highlighting

`POST /api/highlight` accepts the same base64 before/after image fields as prediction. It validates and orients the images, runs the checksum-verified frozen segmenter on CPU, and returns JPEG overlay data URLs plus model revision and decoded-image hashes. It does not save uploaded images or infer grams. Rendering uses the same unedited argmax mask and 35% blue overlay as the sample view. A shared inference lock bounds simultaneous encoder/segmenter execution.

Install `requirements-segmentation.txt` and use the pinned checkpoint setup above. On this workspace, run `.venv-segmentation/bin/python -m foodvision demo --port 8765`. Missing optional dependencies or weights produce a 503 with setup guidance; malformed uploads produce a 400. The percentage endpoint remains separate. Browser state invalidates prior masks when either image changes and discards stale mask responses.

A real local API smoke test on uploaded L492 JPEGs returned both masks in 2.33 seconds on this machine; that single observation is not a latency benchmark. These are the same imperfect food masks, not improved residue recognition. The L133 cached masks cover 13.9% of the before image and 36.8% of the after image, despite a recorded remaining fraction of zero. This illustrates why area is not interchangeable with mass.

See [the research comparison](RELATED-WORK.md) for the next direction: separately evaluate residue recognition and portion geometry instead of treating more regression tuning as a guaranteed solution.

## Relative-depth follow-up

The subsequent [depth experiment](DEPTH.md) evaluated Depth Anything V2 Small features over all 524 LeFood pairs with the original nested splits. Change + depth reached 13.50 pp; Change + mask + depth reached 11.99 pp, versus 9.63 pp for the current estimator. Neither passed the deployment gates. The report includes matched no-depth controls and failures on recorded-zero/small-positive slices. It is a frozen-feature experiment, not a trained RGB/depth fusion model or a measured-volume estimator.
