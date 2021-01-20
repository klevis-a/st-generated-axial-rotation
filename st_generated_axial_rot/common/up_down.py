from typing import NamedTuple
import numpy as np
from biokinepy.np_utils import find_runs
from biokinepy.trajectory import PoseTrajectory


class UpDownAnalysis(NamedTuple):
    """ Class containing analysis results for where minimum and maximal HT elevations are achieved.

    Attributes
    ----------
    max_elev: float
        Maximum HT elevation (deg or rad).
    max_elev_idx: int
        Frame index where maximum HT elevation occurs.
    min_elev_up: float
        Minimal HT elevation (deg or rad) during arm elevation.
    min_elev_down: float
        Minimal HT elevation (deg or rad) during arm lowering.
    min_elev_up_idx: int
        Frame index where minimal HT eleation occurs during arm elevation.
    min_elev_down_idx: int
        Frame index where minimal HT elevation occurs during arm lowering.
    max_run_up_start_idx: int
        Starting frame index of the longest stretch of increasing HT elevation during arm elevation.
    max_run_up_end_idx: int
        Ending frame index of the longest stretch of increasing HT elevation during arm elevation.
    max_run_down_start_idx: int
        Starting frame index of the longest stretch of decreasing HT elevation during arm lowering.
    max_run_down_end_idx: int
        Ending frame index of the longest stretch of decreasing HT elevation during arm lowering.
    max_run_up_start_val: float
        HT elevation (deg or rad) at the start of the longest stretch of increasing HT elevation during arm elevation.
    max_run_up_end_val: float
        HT elevation (deg or rad) at the end of the longest stretch of increasing HT elevation during arm elevation.
    max_run_down_start_val: float
        HT elevation (deg or rad) at the start of the longest stretch of decreasing HT elevation during arm lowering.
    max_run_down_end_val: float
        HT elevation (deg or rad) at the end of the longest stretch of decreasing HT elevation during arm lowering.
    """
    max_elev: float
    max_elev_idx: int
    min_elev_up: float
    min_elev_down: float
    min_elev_up_idx: int
    min_elev_down_idx: int
    max_run_up_start_idx: int
    max_run_up_end_idx: int
    max_run_down_start_idx: int
    max_run_down_end_idx: int
    max_run_up_start_val: float
    max_run_up_end_val: float
    max_run_down_start_val: float
    max_run_down_end_val: float


def analyze_up_down(ht_traj: PoseTrajectory) -> UpDownAnalysis:
    """Performing an analysis of ht_traj to determine where minimum and maximal HT elevation is achieved."""
    ht_elev = np.rad2deg(-ht_traj.euler.ht_isb[:, 1])
    num_frames = ht_elev.size
    max_elev = ht_elev.max()
    max_elev_idx = np.argmax(ht_elev)

    # first level of analysis - determine minimum for up and down
    up = ht_elev[:max_elev_idx+1]
    down = np.flip(ht_elev[max_elev_idx:])
    min_elev_up = up.min()
    min_elev_down = down.min()
    min_elev_up_idx = np.argmin(up)
    min_elev_down_idx = np.argmin(down)

    # second level of analysis - determine longest run
    up_diff = np.diff(up)
    down_diff = np.diff(down)
    up_run_vals, up_run_starts, up_run_lengths = find_runs(up_diff >= 0)
    down_run_vals, down_run_starts, down_run_lengths = find_runs(down_diff >= 0)
    up_run_starts_inc = up_run_starts[up_run_vals]
    up_run_lengths_inc = up_run_lengths[up_run_vals]
    down_run_starts_inc = down_run_starts[down_run_vals]
    down_run_lengths_inc = down_run_lengths[down_run_vals]

    # note that the down indices are for the flipped trajectory
    max_run_up_run_idx = np.argmax(up_run_lengths_inc)
    max_run_down_run_idx = np.argmax(down_run_lengths_inc)
    max_run_up_start_idx = up_run_starts_inc[max_run_up_run_idx]
    max_run_up_end_idx = max_run_up_start_idx + up_run_lengths_inc[max_run_up_run_idx]
    max_run_down_start_idx = down_run_starts_inc[max_run_down_run_idx]
    max_run_down_end_idx = max_run_down_start_idx + down_run_lengths_inc[max_run_down_run_idx]

    max_run_up_start_val = up[max_run_up_start_idx]
    max_run_up_end_val = up[max_run_up_end_idx]
    max_run_down_start_val = down[max_run_down_start_idx]
    max_run_down_end_val = down[max_run_down_end_idx]

    return UpDownAnalysis(max_elev, max_elev_idx, min_elev_up, min_elev_down, min_elev_up_idx,
                          num_frames - 1 - min_elev_down_idx, max_run_up_start_idx, max_run_up_end_idx,
                          num_frames - 1 - max_run_down_start_idx, num_frames - 1 - max_run_down_end_idx,
                          max_run_up_start_val, max_run_up_end_val, max_run_down_start_val, max_run_down_end_val)
