# Model and evaluation

The website uses a fixed DINOv2 ViT-S/14 encoder and a regression head calibrated on 50 LeFood pairs. Render runs the encoder through ONNX Runtime on CPU. [Deployment](DEPLOY.md) · [Development](DEVELOPMENT.md).

## Photo processing

Decode the image, apply EXIF orientation, and convert to RGB. Before encoding an upload, reject identical decoded images, constant-color images, and images with either side below 224 pixels. Known flagged starting images also withhold an estimate. [Input validation](RELIABILITY.md).

On Render, the browser first reduces large originals to at most 1600 pixels on the longest side. For accepted server uploads, resize the shorter edge to 256 pixels using bicubic interpolation and take a 224×224 center crop. Normalize with ImageNet channel means and standard deviations. DINOv2 produces a 384-dimensional vector for each photo. Normalize both vectors independently to unit length.

The demo's scalar feature is `1 - dot(before_embedding, after_embedding)`. An RBF kernel regressor predicts the remaining fraction from that change. Calibration-group weights have equal total weight; the target offset is the lower weighted median. The exact bandwidth, regularization, fitted coefficients, training IDs and encoder version are in `models/`.

New-photo predictions run on CPU. The saved example embeddings were generated in the research environment; small backend-related differences are possible when those photographs are encoded again. The included model head reproduces the saved example forecasts from their saved embeddings to numerical precision.

## Compact benchmark

There are 524 LeFood records grouped into 34 food categories and 792 ACETADA records grouped into 152 participants. Splits also separate groups of visually similar images. Every record receives exactly one outer forecast in this benchmark.

Each source has five outer folds. A fixed calibration sample contains 50 calibration labels per fold. Inner grouped partitions select candidates without reading outer labels. The same calibration records are used for all compared methods.

The Appearance bank contains a median control plus 12 after-only and 12 concatenated-pair regressors. The Change bank contains a median control plus 12 vector-change and 12 cosine-change regressors. Each feature uses gamma in `{0.5, 2, 8}` and alpha in `{0.01, 0.1, 1, 10}`. Selection takes the lowest inner group-MAE; candidate order breaks ties. The separate mean control uses group-weighted calibration targets.

The report averages absolute error within original food/participant groups before averaging groups. Errors are reported in percentage points. The benchmark checks selected candidates and forecasts against saved reference predictions, with absolute tolerance `1e-10`.

## Demo examples

Four records were selected at evenly spaced positions in the demo fold's held-out record list. The sample gallery contains three pairs. The website's Accuracy tab includes a fourth rice pair (L81) in the expandable example section. Their food groups and reviewed similarity components are absent from the demo model's 50 calibration records.

Samples use cached embeddings. Accepted uploads run through the encoder and the same regression head.

## Limits

Framing, backgrounds and unfamiliar foods can cause large errors. The benchmark evaluates within-dataset calibration; the datasets have also informed development. A separate serving/session holdout is needed to measure performance on new meals.

The output is a remaining-mass fraction, clipped to 0–1.25. Grams require a known starting mass. The model does not return a calibrated confidence score.
