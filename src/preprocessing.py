"""Public utilities. Fit preprocessing using training data only."""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler


def binary_label_mapping(y_train):
    """Return an ascending-numeric-label mapping to {-1,+1}."""
    labels = np.unique(np.asarray(y_train))
    if labels.size != 2:
        raise ValueError("Exactly two numeric classes are required")
    return {float(labels[0]): -1, float(labels[1]): +1}


def map_labels(y, mapping):
    """Apply the training-derived mapping, rejecting unseen labels."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if not np.all(np.isin(y, list(mapping))):
        raise ValueError("Encountered an unseen class label")
    return np.array([mapping[float(label)] for label in y], dtype=int)


def scale_train_and_test(X_train, X_test):
    """Learn the scaler exclusively on training samples."""
    scaler = StandardScaler().fit(np.asarray(X_train, dtype=float))
    return scaler.transform(X_train), scaler.transform(X_test), scaler
