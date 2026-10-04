# Model and evaluation

The demo is an image-feature regression model, with a fixed DINOv2 ViT-S/14 encoder and a small, locally calibrated regression head.

## Photo processing

Decode the image, apply EXIF orientation, and convert to RGB. Before encoding an upload, reject identical decoded images, constant-color images, and images with either side below 224 pixels. Starting-portion observations are optional; missing or uncertain observations allow inference without claiming food presence was verified. Explicit empty/residue observations supplied through the API and known unresolved starting-image flags still withhold an estimate. These policy checks do not measure food presence or improve the regressor’s benchmark accuracy. [Policy and API contract](RELIABILITY.md).

For accepted uploads, resize the shorter edge to 256 pixels using bicubic interpolation and take a 224×224 center crop. Normalize with ImageNet channel means and standard deviations. DINOv2 produces a 384-dimensional vector for each photo. Normalize both vectors independently to unit length.

The demo's scalar feature is `1 - dot(before_embedding, after_embedding)`. An RBF kernel regressor predicts the remaining fraction from that change. Calibration-group weights have equal total weight; the target offset is the lower weighted median. The exact bandwidth, regularization, fitted coefficients, training IDs and encoder version are in `models/`.

New-photo predictions run on CPU. The saved example embeddings were generated in the research environment; small backend-related differences are possible when those photographs are encoded again. The included model head reproduces the saved example forecasts from their saved embeddings to numerical precision.

## Compact benchmark

There are 524 LeFood records grouped into 34 food categories and 792 ACETADA records grouped into 152 participants. Review-derived image-similarity components provide an additional fitting-separation constraint. Every record receives exactly one outer forecast in this benchmark.

Each source has five outer folds. A fixed calibration sample contains 50 destination labels per fold. Inner grouped partitions select candidates without reading outer labels. The same calibration records are used for all compared methods.

The Appearance bank contains a median control plus 12 after-only and 12 concatenated-pair regressors. The Change bank contains a median control plus 12 vector-change and 12 cosine-change regressors. Each feature uses gamma in `{0.5, 2, 8}` and alpha in `{0.01, 0.1, 1, 10}`. Strict ordered minimum inner group-MAE resolves ties. The separate mean control uses group-weighted calibration targets.

The report averages absolute error within original food/participant groups before averaging groups. It reports percentage-point errors, not classification accuracy. The benchmark checks selected candidates and forecasts against saved reference predictions, with absolute tolerance `1e-10`.

## Demo examples

Four records were selected at evenly spaced positions in the demo fold's held-out record list. The gallery groups them by recorded remaining fraction and includes the largest-error example as a visible challenge case. Their food groups and reviewed similarity components are absent from the demo model's 50 calibration records.

The gallery uses cached embeddings to make the first interaction fast and offline; these historical examples retain their original outputs. An accepted upload invokes the encoder and the same regression head. The exact-image review registry can withhold an upload estimate, but never supplies a prediction. No nearest-image lookup or canned prediction is used for uploads.

## Limits

The benchmark does not perform a fresh dataset download, recompute all image features, retrain CNNs, or measure generalization to an untouched institution. The datasets informed earlier development, so the results support an engineering comparison on these collections. The model is not a validated physical weighing instrument.
