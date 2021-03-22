import numpy as np
from biokinepy.trajectory import PoseTrajectory
from scipy.integrate import cumtrapz
from biokinepy.vec_ops import extended_dot
from biokinepy.vel_acc import ang_vel
from st_generated_axial_rot.common.analysis_utils import generic_add_interp
from st_generated_axial_rot.common.interpolation import ShoulderTrajInterp, interp_vec_traj


def wrt_torso_traj_proj_ext(wrt_torso_traj: np.ndarray, wrt_torso_axes: np.ndarray) -> np.ndarray:
    """Project angular velocity of trajectory that is expressed with respect to torso onto supplied axes
    (also expressed with respect to torso) and integrate from start to end of motion."""
    num_frames = wrt_torso_traj.shape[0]

    # dt doesn't matter because we will integrate soon
    av = ang_vel(wrt_torso_traj, dt=1)

    av_x = (extended_dot(av, wrt_torso_traj[:, :, 0])[..., np.newaxis] * wrt_torso_traj[:, :, 0])
    av_y = (extended_dot(av, wrt_torso_traj[:, :, 1])[..., np.newaxis] * wrt_torso_traj[:, :, 1])
    av_z = (extended_dot(av, wrt_torso_traj[:, :, 2])[..., np.newaxis] * wrt_torso_traj[:, :, 2])

    av_proj_x = extended_dot(av_x, wrt_torso_axes)
    av_proj_y = extended_dot(av_y, wrt_torso_axes)
    av_proj_z = extended_dot(av_z, wrt_torso_axes)
    av_proj = extended_dot(av, wrt_torso_axes)

    induced = np.empty((num_frames, 4), dtype=np.float)
    induced[:, 0] = cumtrapz(av_proj_x, dx=1, initial=0)
    induced[:, 1] = cumtrapz(av_proj_y, dx=1, initial=0)
    induced[:, 2] = cumtrapz(av_proj_z, dx=1, initial=0)
    induced[:, 3] = cumtrapz(av_proj, dx=1, initial=0)

    return induced


def wrt_torso_traj_proj(wrt_torso_traj: np.ndarray, wrt_torso_axes: np.ndarray) -> np.ndarray:
    """Project angular velocity of trajectory that is expressed with respect to torso onto supplied axes
    (also expressed with respect to torso) and integrate from start to end of motion."""
    # dt doesn't matter because we will integrate soon
    av = ang_vel(wrt_torso_traj, dt=1)
    av_proj = extended_dot(av, wrt_torso_axes)

    return cumtrapz(av_proj, dx=1, initial=0)


def gh_traj_proj(gh_traj: np.ndarray, st_traj: np.ndarray, wrt_torso_axes: np.ndarray) -> np.ndarray:
    """Project angular velocity of GH trajectory onto supplied axes (expressed with respect to torso)
    and integrate from start to end of motion."""
    # dt doesn't matter because we will integrate soon
    av = ang_vel(gh_traj, dt=1)
    # express GH angular velocity in torso coordinate system
    av_torso = np.squeeze(st_traj @ av[..., np.newaxis])
    av_proj = extended_dot(av_torso, wrt_torso_axes)

    return cumtrapz(av_proj, dx=1, initial=0)


def compute_elev_axis(ht_traj: np.ndarray) -> np.ndarray:
    """Compute elevation rotation axis for each frame of ht_traj."""
    num_frames = ht_traj.shape[0]
    long_axis = -ht_traj[:, :, 1]
    # compute axis that is perpendicular to the longitudinal axis projection
    long_proj_perp = np.cross(np.tile(np.array([0, 1, 0]), (num_frames, 1)), long_axis)
    long_proj_perp = long_proj_perp / np.sqrt(extended_dot(long_proj_perp, long_proj_perp))[..., np.newaxis]
    return long_proj_perp


def compute_axial_axis(ht_traj: np.ndarray) -> np.ndarray:
    """Compute axial rotation axis for each frame of ht_traj."""
    return ht_traj[:, :, 1]


def compute_poe_axis(ht_traj: np.ndarray) -> np.ndarray:
    """Compute PoE rotation axis for each frame of ht_traj."""
    long_axis = ht_traj[:, :, 1]
    elev_axis = compute_elev_axis(ht_traj)
    poe_axis = np.cross(elev_axis, long_axis)
    poe_axis = poe_axis / np.sqrt(extended_dot(poe_axis, poe_axis))[..., np.newaxis]
    return poe_axis


def add_st_gh_contrib(traj_interp: ShoulderTrajInterp):
    """Add HT, ST, and GH joint contributions for elevation, axial rotation, and POE to traj_interp."""
    st_joint = traj_interp.st
    gh_joint = traj_interp.gh
    ht_joint = traj_interp.ht

    ht_contribs, st_contribs, gh_contribs = add_st_gh_contrib_ind(ht_joint.traj, st_joint.traj, gh_joint.traj)
    ht_joint.contribs = ht_contribs
    st_joint.contribs = st_contribs
    gh_joint.contribs = gh_contribs

    generic_add_interp(traj_interp, 'ht', 'contribs', interp_vec_traj)
    generic_add_interp(traj_interp, 'st', 'contribs', interp_vec_traj)
    generic_add_interp(traj_interp, 'gh', 'contribs', interp_vec_traj)


def add_st_gh_contrib_ind(ht_joint: PoseTrajectory, st_joint: PoseTrajectory, gh_joint: PoseTrajectory):
    """Compute HT, ST, and GH joint contributions for elevation, axial rotation, and PoE given the pose trajectories of
    these joints."""
    # Note that where these contributions are stored differs from the manuscript. To make it easy to compare to ISB
    # PoE changes are stored in index 0, elevation changes are stored in index 1, and axial rotation changes are stored
    # in index 2. The orthogonality of the frame is still respected just the index where these contributions are stored
    # is changed.
    poe_axes = compute_poe_axis(ht_joint.rot_matrix)
    elev_axes = compute_elev_axis(ht_joint.rot_matrix)
    axial_axes = compute_axial_axis(ht_joint.rot_matrix)

    st_contribs = np.empty((st_joint.ang_vel.shape[0], 4))
    st_contribs[:, 0] = wrt_torso_traj_proj(st_joint.rot_matrix, poe_axes)
    st_contribs[:, 1] = wrt_torso_traj_proj(st_joint.rot_matrix, elev_axes)
    st_contribs[:, 2] = wrt_torso_traj_proj(st_joint.rot_matrix, axial_axes)
    st_ang_vel_mag = np.sqrt(extended_dot(st_joint.ang_vel, st_joint.ang_vel))
    st_ang_vel_dir = st_joint.ang_vel / st_ang_vel_mag[..., np.newaxis]
    st_ang_vel_mag = np.where(extended_dot(st_ang_vel_dir, elev_axes) > 0, st_ang_vel_mag, -st_ang_vel_mag)
    st_contribs[:, 3] = cumtrapz(st_ang_vel_mag, dx=st_joint.dt, initial=0)

    ht_contribs = np.empty((ht_joint.ang_vel.shape[0], 4))
    ht_contribs[:, 0] = wrt_torso_traj_proj(ht_joint.rot_matrix, poe_axes)
    ht_contribs[:, 1] = wrt_torso_traj_proj(ht_joint.rot_matrix, elev_axes)
    ht_contribs[:, 2] = wrt_torso_traj_proj(ht_joint.rot_matrix, axial_axes)
    ht_ang_vel_mag = np.sqrt(extended_dot(ht_joint.ang_vel, ht_joint.ang_vel))
    ht_ang_vel_dir = ht_joint.ang_vel / ht_ang_vel_mag[..., np.newaxis]
    ht_ang_vel_mag = np.where(extended_dot(ht_ang_vel_dir, elev_axes) > 0, ht_ang_vel_mag, -ht_ang_vel_mag)
    ht_contribs[:, 3] = cumtrapz(ht_ang_vel_mag, dx=ht_joint.dt, initial=0)

    gh_contribs = np.empty((gh_joint.ang_vel.shape[0], 4))
    gh_contribs[:, 0] = gh_traj_proj(gh_joint.rot_matrix, st_joint.rot_matrix, poe_axes)
    gh_contribs[:, 1] = gh_traj_proj(gh_joint.rot_matrix, st_joint.rot_matrix, elev_axes)
    gh_contribs[:, 2] = gh_traj_proj(gh_joint.rot_matrix, st_joint.rot_matrix, axial_axes)
    gh_ang_vel_mag = np.sqrt(extended_dot(gh_joint.ang_vel, gh_joint.ang_vel))
    gh_ang_vel_dir = (np.squeeze(st_joint.rot_matrix @ gh_joint.ang_vel[..., np.newaxis]) /
                      ht_ang_vel_mag[..., np.newaxis])
    gh_ang_vel_mag = np.where(extended_dot(gh_ang_vel_dir, elev_axes) > 0, gh_ang_vel_mag, -gh_ang_vel_mag)
    gh_contribs[:, 3] = cumtrapz(gh_ang_vel_mag, dx=gh_joint.dt, initial=0)

    return ht_contribs, st_contribs, gh_contribs
