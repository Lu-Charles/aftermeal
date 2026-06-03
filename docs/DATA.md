# Data sources and transformations

## LeFood-Set v1

- Creators: Yuita Arum Sari, Yudi Arimba Wani and Atsushi Nakazawa.
- Dataset: [LeFood-Set: Leftovers Food Dataset](https://data.mendeley.com/datasets/cchsk79jkt/1).
- DOI: `10.17632/cchsk79jkt.1`.
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
- Included: features and remaining-mass labels for 524 pairs, plus four original before/after image pairs for the demo.
- Changes: labels expressed as after-mass divided by before-mass; images embedded using DINOv2. Example photographs are renamed for packaging and cropped to fit the browser display; their file contents are unchanged.

## ACETADA

- Creators: Bruce Coburn, Jiangpeng He, Megan E. Rollo, Satvinder S. Dhaliwal, Deborah A. Kerr and Fengqing Zhu.
- Dataset and citation: [ACETADA release](https://skynet.ecn.purdue.edu/~coburn6/ACETADA/), [associated paper](https://arxiv.org/abs/2507.07048).
- Source license: [Creative Commons Attribution–NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- Included: derived image features, fraction labels and anonymous grouping identifiers for 792 eligible pairs. No original ACETADA photographs are included.
- Changes: image embeddings, eligibility filtering, group-aware partitions and reviewed similarity components. Negative item-weight flags and conflicting duplicate-pair labels are retained in the metadata rather than silently relabeled.

The source data licenses apply to derived data assets in `data/`, photographs and features in `examples/`, and the data-dependent calibration artifact in `models/`; they are not replaced by the code's MIT license. Attribute the datasets when reusing the assets. ACETADA-derived data are for noncommercial use under their source terms.

## Encoder

The demo uses [DINOv2 ViT-S/14](https://github.com/facebookresearch/dinov2) from Meta. Its code revision and pretrained-weight checksum are recorded in `models/demo_model.json`. DINOv2 is separately licensed by its authors; no pretrained encoder weights are redistributed in this package.

## Included inputs

`data/features.npz` stores before and after embeddings and row identifiers. `records.json` holds the matching targets and group identifiers. `episodes.json` defines ten outer evaluation episodes and their inner folds. `reference_predictions.json` stores the comparison forecasts used by the regression check. `checksums.json` verifies these files and the demo model/example assets.

The minimal installation runs the complete included feature benchmark without downloading raw datasets. New-photo inference additionally requires the PyTorch encoder installation and its first-use download.

