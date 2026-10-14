"""Small numeric helpers shared by several modules."""
from __future__ import annotations

import numpy as np


def find_knee(x, y) -> int:
    """Return the index of the 'knee' / 'elbow' of a convex curve.

    Both axes are rescaled to [0, 1]; the knee is the point farthest from the
    straight line joining the first and last points. Works for decreasing curves
    (K-Means inertia) and increasing curves (sorted k-distance plot).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return 0
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    if x_span == 0 or y_span == 0:
        return 0
    xn = (x - x.min()) / x_span
    yn = (y - y.min()) / y_span
    chord = yn[0] + (yn[-1] - yn[0]) * (xn - xn[0]) / (xn[-1] - xn[0])
    return int(np.argmax(np.abs(chord - yn)))
