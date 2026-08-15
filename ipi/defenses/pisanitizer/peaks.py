"""
Peak grouping for PISanitizer.

Port of ``code/defense/PISanitizer-main/PISanitizer/group_peaks.py``.

Turns a per-token attention signal into at most one token span to delete:

  1. Savitzky-Golay smooth (window 5 or 9, polyorder 2).
  2. ``find_peaks`` with a floor of ``height=0.005`` and ``distance=5``.
  3. Join peaks less than ``max_gap`` tokens apart into groups — an injected
     instruction spans many tokens and shows up as a cluster, not a spike.
  4. Drop groups that never reach ``threshold``.
  5. Measure each surviving group's extent with ``peak_widths(rel_height=0.95)``,
     from the left edge of its first peak to the right edge of its last.
  6. **Return only the single highest group.**

Step 6 is easy to miss and it bounds the whole defense: one span per round,
five rounds, so at most five spans are ever removed from a context.

Requires scipy (``pip install ipi-adaptive[pisanitizer]``). Imported lazily so
``import ipi.defenses`` stays dependency-free.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .config import PEAK_PARAMS, SMOOTH_POLYORDER


def group_consecutive_peaks(peaks: Sequence[int], max_gap: int = 20) -> List[List[int]]:
    """Split peak indices into runs separated by more than ``max_gap``."""
    if len(peaks) == 0:
        return []
    groups: List[List[int]] = []
    current = [peaks[0]]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] <= max_gap:
            current.append(peaks[i])
        else:
            groups.append(current)
            current = [peaks[i]]
    groups.append(current)
    return groups


def group_peaks(
    x: Sequence[float],
    smooth_win: int = 9,
    max_gap: int = 10,
    threshold: float = 0.01,
    prominence: float = None,
    distance: int = None,
    height: float = None,
    rel_height: float = None,
) -> Tuple[List[float], List[Tuple[int, int]]]:
    """
    Return ``(smoothed_signal, spans)`` where ``spans`` holds at most one
    inclusive ``(start, end)`` index pair into ``x``.

    Peak-finding parameters default to :data:`~.config.PEAK_PARAMS`.
    """
    import numpy as np
    from scipy.signal import find_peaks, peak_widths, savgol_filter

    prominence = PEAK_PARAMS["prominence"] if prominence is None else prominence
    distance = PEAK_PARAMS["distance"] if distance is None else distance
    height = PEAK_PARAMS["height"] if height is None else height
    rel_height = PEAK_PARAMS["rel_height"] if rel_height is None else rel_height

    x = list(x)
    if len(x) < smooth_win:
        smooth_x = x
    else:
        smooth_x = list(savgol_filter(x, smooth_win, SMOOTH_POLYORDER,
                                      mode="constant", cval=0.0))

    peaks, _ = find_peaks(smooth_x, prominence=prominence, distance=distance,
                          height=height)
    peak_groups = group_consecutive_peaks(peaks, max_gap=max_gap)
    above = {i for i, v in enumerate(smooth_x) if v >= threshold}

    spans: List[Tuple[int, int]] = []
    top_values: List[float] = []
    for group in peak_groups:
        if not (set(group) & above):
            continue
        widths = peak_widths(smooth_x, group, rel_height=rel_height)
        left_ips, right_ips = widths[-2], widths[-1]
        spans.append((int(left_ips[0]), int(right_ips[-1])))
        top_values.append(max(smooth_x[j] for j in group))

    if not top_values:
        return smooth_x, []

    # Only the strongest group is removed per round.
    return smooth_x, [spans[int(np.argmax(top_values))]]
