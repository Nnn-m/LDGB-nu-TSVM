"""Protocol utilities only; does not execute the paper's full benchmarks."""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def outer_splits(X, y, random_state=42):
    """10 repetitions of stratified 10-fold CV."""
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=random_state)
    yield from cv.split(X, y)


def inner_splits(X_outer_train, y_outer_train, random_state=42):
    """Hyperparameters must be tuned on the outer-training partition only."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    yield from cv.split(X_outer_train, y_outer_train)


def corrupt_training_labels(y_train, ratio, random_state=42):
    """Flip a per-class fraction of +/-1 training labels; never touch test labels."""
    if not 0 <= ratio <= 1:
        raise ValueError("ratio must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    labels = np.asarray(y_train, dtype=int).copy()
    if not np.all(np.isin(labels, [-1, 1])):
        raise ValueError("labels must be mapped to {-1, +1}")
    original = labels.copy()
    for klass in (-1, 1):
        indices = np.flatnonzero(original == klass)
        n_flip = int(round(len(indices) * ratio))
        selected = rng.choice(indices, size=n_flip, replace=False)
        labels[selected] *= -1
    return labels
