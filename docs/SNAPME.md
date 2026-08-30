# SNAPMe fit assessment

Based on primary-source descriptions; no dataset archive downloaded, imported or evaluated here.

The [SNAPMe paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10708545/) describes 3,311 photographs from 95 participants, including 1,475 before and 1,436 after photographs. Capture instructions included a known-size marker. Labels link images with ASA24 diet records. The custom after-photo entries received zero nutrient values to avoid duplicate annotation; those values are **not evidence of zero leftover mass**. Missing companion photographs must also remain missing rather than become empty-plate examples.

The [USDA dataset record](https://agdatacommons.nal.usda.gov/articles/dataset/SNAPMe_A_Benchmark_Dataset_of_Food_Photos_with_Food_Records_for_Evaluation_of_Computer_Vision_Algorithms_in_the_Context_of_Dietary_Assessment/24856449) explicitly recommends benchmark evaluation rather than training on this small curated dataset. The archive is approximately 1.89 GB. The [author repository](https://github.com/JulesLarke-USDA/SNAPMe) supplies analysis code and ingredient-prediction evaluation materials.

## Proposed use in Aftermeal

Keep SNAPMe outside model training and threshold tuning. Define the test before viewing outcomes, and group related images by participant/day/meal. Match each before photograph with its documented after filename, explicitly recording missing companions and excluding packaging-label pairs from the meal-pair task.

Use the existing reference descriptions to test food/ingredient recognition. A separate, independently annotated visual subset could test visible portions, residue, clean plates and ambiguous cases. New visual annotations would support only that visual task; they would not establish weighed leftover ratios. No mass-error number should be reported without a suitable measured mass target.

A small vision-language model could produce structured proposed material labels and flag input problems. Evaluate those outputs against independent labels, including its tendency to miss tiny real portions. Record label provenance and abstentions. Do not use its own proposed labels as its test answers, and do not treat self-reported model confidence as calibrated reliability.

The role split is an engineering proposal: a food-mass dataset can support training portion estimation, while SNAPMe can test transfer to participant-taken photographs. The current depth experiment has not established that either a language-model gate or a depth blend improves accuracy.
