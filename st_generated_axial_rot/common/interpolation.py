import numpy as np
import quaternion as q
from biokinepy.trajectory import PoseTrajectory
from st_generated_axial_rot.common.up_down import UpDownAnalysis


def set_long_axis_hum(traj: PoseTrajectory) -> None:
    """Set the longitudinal axis on humeral trajectory according to ISB specifications."""
    traj.long_axis = np.array([0, 1, 0])


def set_long_axis_scap(traj: PoseTrajectory) -> None:
    """Set the third axis of rotation on the scapula trajectory according to ISB specifications."""
    traj.long_axis = np.array([0, 0, 1])


class ShoulderTrajInterp:
    """Interpolated shoulder trajectory.

    Attributes
    ----------
    trial_name: str
        Trial name which contains this interpolated sholder trajectory.
    ht: biokinepy.trajectory.PoseTrajectory
        HT trajectory.
    st: biokinepy.trajectory.PoseTrajectory
        ST trajectory.
    gh: biokinepy.trajectory.PoseTrajectory
        GH trajectory.
    up_down_idx: axial_rot_analysis.common.up_down.UpDownAnalysis
        Up down analysis object for this trial.
    common_ht_endpts: numpy.ndarrary
        Start and end HT elevation (deg or rad based on values in up_down_idx) for which interpolation should occur
        (this will be the same for all trials).
    coarse_dtheta: float
        Delta (deg or rad based on values in up_down_idx) for coarse interpolation.
    fine_dtheta: float
        Delta (deg or rad based on values in up_down_idx) for fine interpolation.
    common_ht_range_coarse: numpy.ndarray
        Numpy array containing HT elevations at which coarse interpolation should occur to overlap all trajectories
        (based on common_ht_endpts and coarse_dtheta).
    common_ht_range_fine: numpy.ndarray
        Numpy array containing HT elevations at which fine interpolation should occur to overlap all trajectories
        (based on common_ht_endpts and fine_dtheta).
    sym_ht_endpts: List[float]
        Starting and ending HT elevations that encompass both arm raising and lowering for this trial (trial-specific).
    sym_ht_range_fine: np.ndarra
        Numpy array containing HT elevations at which fine interpolation should occur to overlap both arm raising and
        lowering for this trial (based on sym_ht_endpts and fine_dtheta).
    """
    def __init__(self, trial_name: str, ht: PoseTrajectory, gh: PoseTrajectory, st: PoseTrajectory,
                 up_down_idx: UpDownAnalysis, fine_dtheta: float, coarse_dtheta: float, common_ht_endpts: np.ndarray):
        set_long_axis_hum(ht)
        set_long_axis_hum(gh)
        set_long_axis_scap(st)
        self.trial_name = trial_name
        self.up_down_idx = up_down_idx
        self.fine_dtheta = fine_dtheta
        self.coarse_dtheta = coarse_dtheta
        self.ht_ea = np.rad2deg(-ht.euler.ht_isb[:, 1])
        self.ht_ea_up = self.get_up(self.ht_ea)
        self.ht_ea_down = self.get_down(self.ht_ea)
        self.common_ht_endpts = common_ht_endpts
        self.common_ht_range_fine = np.arange(self.common_ht_endpts[0], self.common_ht_endpts[1] + self.fine_dtheta,
                                              self.fine_dtheta)
        self.common_ht_range_coarse = np.arange(self.common_ht_endpts[0], self.common_ht_endpts[1] + self.coarse_dtheta,
                                                self.coarse_dtheta)

        # calculate symmetric up down humerothoracic range - used for comparing up/down
        sym_ht_start = np.ceil(max(self.up_down_idx.max_run_up_start_val, self.up_down_idx.max_run_down_start_val))
        sym_ht_end = np.floor(min(self.up_down_idx.max_run_up_end_val, self.up_down_idx.max_run_down_end_val))
        self.sym_ht_endpts = [sym_ht_start, sym_ht_end]
        self.sym_ht_range_fine = np.arange(sym_ht_start, sym_ht_end + self.fine_dtheta, self.fine_dtheta)

        self.ht = JointTrajInterp(ht, self, long_axis_fnc=set_long_axis_hum)
        self.gh = JointTrajInterp(gh, self, long_axis_fnc=set_long_axis_hum)
        self.st = JointTrajInterp(st, self, long_axis_fnc=set_long_axis_scap)

    def get_up(self, measure: np.ndarray) -> np.ndarray:
        """Return up section of the trajectory specified in measure (M, N), where M is number of frames."""
        return measure[self.up_down_idx.max_run_up_start_idx:self.up_down_idx.max_run_up_end_idx + 1]

    def get_down(self, measure: np.ndarray) -> np.ndarray:
        """Return down section of the trajectory specified in measure (M, N), where M is number of frames."""
        return np.flip(measure[self.up_down_idx.max_run_down_end_idx:self.up_down_idx.max_run_down_start_idx + 1])


class JointTrajInterp:
    """An interpolate joint trajectory.

    Attributes
    ----------
    traj: biokinepy.trajectory.PoseTrajectory
        The PoseTrajectory to be interpolated.
    up: biokinepy.trajectory.PoseTrajectory
        Arm raising portion of trajectory.
    down: biokinepy.trajectory.PoseTrajectory
        Arm lowering portion of trajectory.
    sym_fine_up: biokinepy.trajectory.PoseTrajectory
        Finely interpolated arm raising trajectory encompassing a HT elevation region that is present for both the arm
        raising and lowering phases of this trajectory.
    sym_fine_down: biokinepy.trajectory.PoseTrajectory
        Finely interpolated arm lowering trajectory encompassing a HT elevation region that is present for both the arm
        raising and lowering phases of this trajectory.
    common_fine_up: biokinepy.trajectory.PoseTrajectory
        Finely interpolated arm raising trajectory encompassing a HT elevation that is present for all trials during
        arm raising.
    common_fine_down: biokinepy.trajectory.PoseTrajectory
        Finely interpolated arm lowering trajectory encompassing a HT elevation that is present for all trials during
        arm lowering.
    common_coarse_up: biokinepy.trajectory.PoseTrajectory
        Coarsely interpolated arm raising trajectory encompassing a HT elevation that is present for all trials during
        arm raising.
    common_coarse_down: biokinepy.trajectory.PoseTrajectory
        Coarsely interpolated arm lowering trajectory encompassing a HT elevation that is present for all trials during
        arm lowering.
    true_axial_rot: np.ndarray
        True axial rotation for this trajectory.
    true_axial_rot_up: np.ndarray
        True axial rotation for the arm raising portion of this trajectory.
    true_axial_rot_down: np.ndarray
        True axial rotation for the arm lowering portion of this trajectory.
    true_axial_rot_sym_fine_up: np.ndarray
        Finely interpolated true axial rotation during arm raising encompassing a HT elevation region that is present
        for both the arm raising and lowering phases of this trajectory.
    true_axial_rot_sym_fine_down: np.ndarray
        Finely interpolated true axial rotation during arm lowering encompassing a HT elevation region that is present
        for both the arm raising and lowering phases of this trajectory.
    true_axial_rot_common_fine_up: np.ndarray
        Finely interpolated true axial rotation during arm raising encompassing a HT elevation that is present for all
        trials during arm raising.
    true_axial_rot_common_fine_down: np.ndarray
        Finely interpolated true axial rotation during arm lowering encompassing a HT elevation that is present for all
        trials during arm lowering.
    true_axial_rot_common_coarse_up: np.ndarray
        Coarsely interpolated true axial rotation during arm raising encompassing a HT elevation that is present for all
        trials during arm raising.
    true_axial_rot_common_coarse_down: np.ndarray
        Coarsely interpolated true axial rotation during arm lowering encompassing a HT elevation that is present for
        all trials during arm lowering.
    """
    def __init__(self, traj: PoseTrajectory, shoulder_traj: ShoulderTrajInterp, long_axis_fnc):
        self.traj = traj
        self.shoulder_traj = shoulder_traj
        pos = np.zeros((self.traj.quat.size, 3))
        pos_up = shoulder_traj.get_up(pos)
        pos_down = shoulder_traj.get_down(pos)
        quat_up = shoulder_traj.get_up(self.traj.quat)
        quat_down = shoulder_traj.get_down(self.traj.quat)
        self.up = PoseTrajectory.from_quat(pos_up, q.as_float_array(quat_up))
        self.down = PoseTrajectory.from_quat(pos_down, q.as_float_array(quat_down))
        self.sym_fine_up = interp_quat_traj(shoulder_traj.ht_ea_up, quat_up, shoulder_traj.sym_ht_range_fine)
        self.sym_fine_down = interp_quat_traj(shoulder_traj.ht_ea_down, quat_down, shoulder_traj.sym_ht_range_fine)
        self.common_fine_up = interp_quat_traj(shoulder_traj.ht_ea_up, quat_up, shoulder_traj.common_ht_range_fine)
        self.common_fine_down = interp_quat_traj(shoulder_traj.ht_ea_down, quat_down,
                                                 shoulder_traj.common_ht_range_fine)
        self.common_coarse_up = interp_quat_traj(shoulder_traj.ht_ea_up, quat_up, shoulder_traj.common_ht_range_coarse)
        self.common_coarse_down = interp_quat_traj(shoulder_traj.ht_ea_down, quat_down,
                                                   shoulder_traj.common_ht_range_coarse)

        long_axis_fnc(self.up)
        long_axis_fnc(self.down)
        long_axis_fnc(self.sym_fine_up)
        long_axis_fnc(self.sym_fine_down)
        long_axis_fnc(self.common_fine_up)
        long_axis_fnc(self.common_fine_down)
        long_axis_fnc(self.common_coarse_up)
        long_axis_fnc(self.common_coarse_down)

        # compute axial rotation interpolation - this must be done separately than the computations above because
        # axial rotation is path dependent
        self.true_axial_rot = traj.true_axial_rot
        self.true_axial_rot_up = shoulder_traj.get_up(self.true_axial_rot)
        self.true_axial_rot_down = shoulder_traj.get_down(self.true_axial_rot)
        self.true_axial_rot_sym_fine_up = interp_scalar_traj(shoulder_traj.ht_ea_up, self.true_axial_rot_up,
                                                             shoulder_traj.sym_ht_range_fine)
        self.true_axial_rot_sym_fine_down = interp_scalar_traj(shoulder_traj.ht_ea_down, self.true_axial_rot_down,
                                                               shoulder_traj.sym_ht_range_fine)
        self.true_axial_rot_common_fine_up = interp_scalar_traj(shoulder_traj.ht_ea_up, self.true_axial_rot_up,
                                                                shoulder_traj.common_ht_range_fine)
        self.true_axial_rot_common_fine_down = interp_scalar_traj(shoulder_traj.ht_ea_down, self.true_axial_rot_down,
                                                                  shoulder_traj.common_ht_range_fine)
        self.true_axial_rot_common_coarse_up = interp_scalar_traj(shoulder_traj.ht_ea_up, self.true_axial_rot_up,
                                                                  shoulder_traj.common_ht_range_coarse)
        self.true_axial_rot_common_coarse_down = interp_scalar_traj(shoulder_traj.ht_ea_down, self.true_axial_rot_down,
                                                                    shoulder_traj.common_ht_range_coarse)


def interp_scalar_traj(x_orig: np.ndarray, scalar_orig: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return interpolated scalar trajectory with NaNs outside of region defined in x_orig."""
    return np.interp(x, x_orig, scalar_orig, left=np.nan, right=np.nan)


def interp_vec_traj(x_orig: np.ndarray, vec_orig: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return interpolated (M, N) vector trajectory - M timepoints, N dimsions - with NaNs outside of region defined in
    x_orig."""
    interp_vec_traj = np.empty((x.size, vec_orig.shape[1]), dtype=vec_orig.dtype)
    for i in range(vec_orig.shape[1]):
        interp_vec_traj[:, i] = np.interp(x, x_orig, vec_orig[:, i], left=np.nan, right=np.nan)
    return interp_vec_traj


def interp_quat_traj(x_orig: np.ndarray, quat_orig: np.ndarray, x: np.ndarray) -> PoseTrajectory:
    """Return interpolated biokinepy.trajectory.PoseTrajectory (SLERP) with NaNs outside of region defined in x_orig.

    NOTE: only orientation is interpolated, and the position data is instantiated with zeros."""
    quat_interp = interp_traj_slerp(x_orig, quat_orig, x)
    return PoseTrajectory.from_quat(np.zeros((quat_interp.size, 3)), q.as_float_array(quat_interp))


def interp_traj_slerp(x_orig: np.ndarray, quat_orig: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Use SLERP to interpolate quaternion trajectory returning NaNs outside of region specfied in x_orig."""
    # find at which indices the desired x would have occurred
    idx_interp = np.interp(x, x_orig, np.arange(x_orig.size), left=np.nan, right=np.nan)
    # now perform quaternion interpolation
    quat_interp = np.empty(x.size, dtype=np.quaternion)
    for n, x_idx in enumerate(idx_interp):
        if np.isnan(x_idx):
            quat_interp[n] = q.from_float_array([np.nan, np.nan, np.nan, np.nan])
        else:
            t1 = int(np.floor(x_idx))
            t2 = int(np.ceil(x_idx))
            r1 = quat_orig[t1]
            r2 = quat_orig[t2]
            quat_interp[n] = q.slerp(r1, r2, t1, t2, x_idx)
    return quat_interp
