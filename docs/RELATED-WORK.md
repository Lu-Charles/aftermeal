# Related work

These papers informed the segmentation and depth experiments. Their metrics use different datasets and targets.

| Work | What the researchers did | Relevance and limits for Aftermeal |
|---|---|---|
| [Nutrition5k, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf) | Used overhead RGB-D, food segmentation and calibrated capture geometry to derive a volume estimate, then learned mass from RGB plus volume. Table 4 reports 38.1 g MAE for image-only and 29.4 g with volume. | Test geometry as an additional signal, not just mask area. This is total dish mass, not a before/after leftover ratio; their depth was measured by a sensor. |
| [DPF-Nutrition, Foods 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10706621/) | Predicted depth with DPT and fused RGB/depth features using multiscale processing and cross-modal attention. Its Table 7 ablation reports mass MAE of 27.2 g for the RGB stream and 21.2 g for the full model. | Predicted depth is a testable option when users lack depth hardware. The method trains RGB/depth fusion with food-mass supervision. |
| [Food Portion Estimation via 3D Object Scaling, CVPR Workshops 2024](https://arxiv.org/abs/2404.12257) | Used physical scene references, camera/object pose and known 3D food models to estimate volume from a 2D image. | Establish scale and geometry. Dependence on suitable 3D food models limits direct application to mixed dishes, smears and arbitrary leftovers. |
| [Food Waste Detection in Canteen Plates Using YOLOv11, 2025](https://www.mdpi.com/2076-3417/15/13/7137) | Trained instance segmentation in a controlled canteen setting, separating plate/item regions and using area-based waste calculations. Reports mAP 0.343, precision 0.62 and recall 0.322 in its conclusion. | Plate localization and item-specific masks are useful ideas. Its reported metrics evaluate detection and segmentation. |
| [FLIC, Applied Sciences 2026](https://www.mdpi.com/2076-3417/16/11/5465) | Paired full/leftover images, pixel annotations, physically measured masses and fixed camera geometry. Uses digital density to relate apparent area to mass. Includes residue and inedible remnants in its stated leftover definition. | Match the capture protocol, target definition and annotation quality. The capture protocol and residue definition are relevant to future data collection. |

The [official Nutrition5k repository](https://github.com/google-research-datasets/Nutrition5k) supplies RGB/depth data and ingredient/dish masses, allows individual file downloads, and keeps incremental scans of a plate in the same split. Its full archive is 181.4 GB, so any feasibility experiment should start with a bounded subset rather than the full download. It is not a post-consumption paired dataset.

## Next experiments

The current segmenter often includes thin residue, while the regressor uses global image features without explicit geometry. Three useful follow-ups are:

1. Label residue and small portions separately, then evaluate the segmenter on those cases.
2. Compare appearance, mask area and depth under the same serving/session splits.
3. Test the selected model on new weighed servings with varied foods and camera angles.

The completed [frozen-depth comparison](DEPTH.md) did not improve the deployed estimator.
