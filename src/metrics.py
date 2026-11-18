"""Explicit F1 variants; do not conflate them in manuscript reporting."""
from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score


def evaluate_binary(y_true, y_pred, minority_label=None):
    """Return accuracy (%), +1 F1, macro-F1, and optionally minority F1."""
    out = {
        "ACC_percent": 100.0 * accuracy_score(y_true, y_pred),
        "F1_positive": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, labels=[-1, 1], average="macro", zero_division=0),
    }
    if minority_label is not None:
        out["F1_minority"] = f1_score(y_true, y_pred, pos_label=minority_label, zero_division=0)
    return out
