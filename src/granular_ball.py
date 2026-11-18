"""Documented algorithm interface only. Full construction is not public here."""
from dataclasses import dataclass
import numpy as np

@dataclass
class GranularBall:
    center: np.ndarray
    radius: float
    label: int
    purity: float


def build_granular_balls(X_train, y_train, *, k, n_min, tau_p, tau_d, omega_min):
    """Construct locally weighted granular balls from training-only samples.

    Conceptual stages: class-conditional local reliability -> bounded sample
    weights -> representative center/radius -> purity/density-based splitting.
    The paper-specific weighting formula, split decisions and stopping rules
    are not included in this partial public release.
    """
    raise NotImplementedError("Granular-ball implementation not part of partial public release")
