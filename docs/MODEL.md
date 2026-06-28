# Model and evaluation

The target is remaining food mass divided by starting food mass. Frozen DINOv2
embeddings are normalized independently, then represented as After, Paired,
Vector change or Cosine change features. Group-weighted kernel regressors
use inner-only parameter selection with equal-sized candidate banks.

Five held-out folds per source use 50 calibration pairs. Groups and similarity
components remain separate. Predictions are bounded to 0–1.25 for evaluation.
This is within-collection evaluation, not a transfer or physical-volume claim.
