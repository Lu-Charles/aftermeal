# SNAPMe dataset assessment

SNAPMe could support food-recognition evaluation, but it lacks the weighed leftover ratios needed for Aftermeal’s mass-fraction benchmark.

The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10708545/) describes 3,311 photos from 95 participants, including 1,475 before and 1,436 after photos. Images link to ASA24 diet records. After-photo nutrient entries were set to zero to avoid duplicate annotation; they are not leftover-weight measurements.

The [USDA record](https://agdatacommons.nal.usda.gov/articles/dataset/SNAPMe_A_Benchmark_Dataset_of_Food_Photos_with_Food_Records_for_Evaluation_of_Computer_Vision_Algorithms_in_the_Context_of_Dietary_Assessment/24856449) recommends evaluation rather than training. The [author repository](https://github.com/JulesLarke-USDA/SNAPMe) includes analysis code and ingredient-prediction evaluation materials.

A future evaluation could group related photos by participant, day and meal, match documented before/after pairs, and test food recognition against the reference descriptions. Missing companions and packaging photos would need separate handling.

SNAPMe has not been imported into this project.
