# Algorithm overview (not executable full implementation)

**Inputs:** features X, binary labels y, local-density and splitting settings, ν-TSVM settings.

1. Within each training fold, standardize the features using training statistics only and map labels to {-1,+1}.
2. Estimate class-conditional local outlier information and assign bounded local-density weights.
3. Form granular-ball summaries with center, radius, majority label, purity, and density information.
4. Recursively split eligible balls subject to the paper's purity/density and minimum-size conditions.
5. Form two kernelized ν-TSVM quadratic programs from the resulting ball representations.
6. Solve the programs and evaluate held-out examples by the twin-hyperplane decision function.
7. Use fold-isolated hyperparameter optimization and clean outer-fold testing.

The full weighting routine, granular-ball splitting implementation, optimization matrices, solver settings, and complete evaluation scripts are **not included here**. Consult the associated manuscript for formal definitions. This is a *partial research framework*, not a claim of full numerical reproducibility.
