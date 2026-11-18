# LDGB-ν-TSVM: Research Framework (Partial Release)

This repository provides a **partial research framework** related to the manuscript on local-density-weighted granular-ball ν-twin support vector machines. It documents the main computational pipeline, public preprocessing and evaluation utilities, and the experimental protocol.

**Availability and limitations.** This is **not a complete implementation** of LDGB-ν-TSVM and **cannot reproduce the manuscript's numerical tables**. In particular, the local-density-weighted granular-ball construction and the twin-hyperplane quadratic-programming training and prediction routines are represented only by documented interfaces, not executable implementations. The experimental module demonstrates partitioning and training-only label corruption; it does not execute the full model, benchmark baselines, or nested parameter search.

## Workflow

1. Map binary labels to -1/+1 and fit the scaler on each training fold only.
2. Compute class-conditional local reliability and construct granular balls (implementation **not included**).
3. Train kernelized ν-TSVM on granular-ball representations (implementation **not included**).
4. Predict held-out examples and evaluate classification performance.
5. For the manuscript's main experiments, independently select hyperparameters with stratified inner 3-fold CV within each of 10 repeats × 10 outer stratified folds. Do not tune on the held-out outer test folds.

## Included code

- `src/preprocessing.py`: binary-label mapping, leakage-safe feature scaling.
- `src/metrics.py`: accuracy, positive-class F1, macro-F1, minority-class F1 utilities. Confirm which metric produced each manuscript table from its original experiment script before claiming numerical reproducibility.
- `src/granular_ball.py`: data structure and documented interface; **not implemented**.
- `src/ldgb_nu_tsvm.py`: documented model interface; **not implemented**.
- `experiments/experiment_protocol.py`: outer/inner split generators and training-only symmetric label corruption.
- `experiments/parameter_settings.json`: protocol settings and parameter names; not a record of selected fold-specific configurations.
- `tests/test_public_utilities.py`: unit tests for the included public utilities.

## Environment

Python 3.9, NumPy 2.0.2, pandas 2.2.3, CVXOPT 1.3.2, scikit-learn 1.6.1. The `requirements.txt` lists those manuscript-reported versions. The public utilities do not require running the unpublished QP solver.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Citation and code availability

If you use this framework, please cite the associated manuscript once its bibliographic details are available. The full research implementation and exact result-producing experiment scripts are **not included in this public framework**. Any additional provision of code is subject to the authors' sharing decision and the journal's editorial requirements.
