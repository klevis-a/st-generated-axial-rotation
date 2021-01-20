from typing import Union, Tuple, Sequence
import pandas as pd
import numpy as np
import quaternion as q
from biokinepy.cs import ht_inv
from biokinepy.segment_cs import scapula_cs_isb
from biokinepy.trajectory import PoseTrajectory
from biokinepy.vec_ops import extended_dot
from scipy.integrate import cumtrapz
from st_generated_axial_rot.common.database import trajectories_from_trial, BiplaneViconTrial
from st_generated_axial_rot.common.interpolation import ShoulderTrajInterp, interp_vec_traj
from st_generated_axial_rot.common.up_down import analyze_up_down
from st_generated_axial_rot.common.python_utils import rgetattr
import logging

log = logging.getLogger(__name__)


def get_trajs(trial: BiplaneViconTrial, dt: float, torso_def: str, use_ac: bool = False) \
        -> Tuple[PoseTrajectory, PoseTrajectory, PoseTrajectory]:
    """Return HT, GH, and ST trajectory from a given trial."""
    torso, scap, hum = trajectories_from_trial(trial, dt, torso_def=torso_def)
    # this is not efficient because I recompute the scapula CS for each trial, but it can be done at the subject level
    # however, in the grand scheme of things, this computation is trivial
    if use_ac:
        scap_gc = scapula_cs_isb(trial.subject.scapula_landmarks['GC'], trial.subject.scapula_landmarks['IA'],
                                 trial.subject.scapula_landmarks['TS'])
        scap_ac = scapula_cs_isb(trial.subject.scapula_landmarks['AC'], trial.subject.scapula_landmarks['IA'],
                                 trial.subject.scapula_landmarks['TS'])
        scap_gc_ac = ht_inv(scap_gc) @ scap_ac
        scap = PoseTrajectory.from_ht(scap.ht @ scap_gc_ac[np.newaxis, ...], dt=scap.dt, frame_nums=scap.frame_nums)

    ht = hum.in_trajectory(torso)
    gh = hum.in_trajectory(scap)
    st = scap.in_trajectory(torso)
    return ht, gh, st


def extract_sub_rot(shoulder_traj: ShoulderTrajInterp, traj_def: str, y_def: str, decomp_method: str,
                    sub_rot: Union[int, None]) -> np.ndarray:
    """Extract the specified subrotation given an interpolated shoulder trajectory (shoulder_traj),
    joint (traj_def, e.g. ht, gh, st), interpolation (y_def, e.g. common_fine_up),
    decomposition method (decomp_method, e.g. euler.ht_isb), and subrotation (sub_rot, e.g. 0, 1, 2, None)."""
    # first extract ht, gh, or st
    joint_traj = getattr(shoulder_traj, traj_def)
    # Then extract the decomp_method. Note that each JointTrajectory actually computes separate scalar interpolation for
    # true_axial_rot (that's why I don't access the PoseTrajectory below) because true_axial_rot is path dependent so
    # it doesn't make sense to compute it on a trajectory that starts at 25 degrees (for example)
    if decomp_method == 'true_axial_rot':
        y = getattr(joint_traj, 'axial_rot_' + y_def)
    elif decomp_method == 'induced_axial_rot':
        y = getattr(joint_traj, 'induced_axial_rot_' + y_def)[:, sub_rot]
    else:
        y = rgetattr(getattr(joint_traj, y_def), decomp_method)[:, sub_rot]
    return y


def extract_sub_rot_norm(shoulder_traj: ShoulderTrajInterp, traj_def: str, y_def: str, decomp_method: str,
                         sub_rot: Union[int, None], norm_by: str) -> np.ndarray:
    """Extract and normalize the specified subrotation given an interpolated shoulder trajectory (shoulder_traj),
    joint (traj_def, e.g. ht, gh, st), interpolation (y_def, e.g. common_fine_up),
    decomposition method (decomp_method, e.g. euler.ht_isb), subrotation (sub_rot, e.g. 0, 1, 2, None), and
    normalization section (norm_by, e.g. traj, up, down, ...)."""
    y = extract_sub_rot(shoulder_traj, traj_def, y_def, decomp_method, sub_rot)
    # first extract ht, gh, or st
    joint_traj = getattr(shoulder_traj, traj_def)
    if decomp_method == 'true_axial_rot':
        if norm_by == 'traj':
            y0 = getattr(joint_traj, 'axial_rot')[0]
        else:
            y0 = getattr(joint_traj, 'axial_rot_' + norm_by)[0]
    elif decomp_method == 'induced_axial_rot':
        if norm_by == 'traj':
            y0 = getattr(joint_traj, 'induced_axial_rot')[0, sub_rot]
        else:
            y0 = getattr(joint_traj, 'induced_axial_rot_' + norm_by)[0, sub_rot]
    else:
        y0 = rgetattr(getattr(joint_traj, norm_by), decomp_method)[0, sub_rot]

    return y-y0


def ht_min_max(df_row: pd.Series) -> Tuple[float, float, float, int]:
    """Compute minimum HT elevation during elevation and depression, as well as maximum HT elevation and the frame index
    where it occurs."""
    max_elev = df_row['ht'].euler.ht_isb[:, 1].min()
    max_elev_idx = np.argmin(df_row['ht'].euler.ht_isb[:, 1])
    min_elev_elev = df_row['ht'].euler.ht_isb[:max_elev_idx+1, 1].max()
    min_elev_depress = df_row['ht'].euler.ht_isb[max_elev_idx:, 1].max()
    return min_elev_elev, min_elev_depress, max_elev, max_elev_idx


def prepare_db(db: pd.DataFrame, torso_def: str, use_ac: bool, dtheta_fine: float, dtheta_coarse: float,
               ht_endpts_common: Union[Sequence, np.ndarray], should_fill: bool = True,
               should_clean: bool = True) -> None:
    """Prepared database for analysis by computing HT, ST, and GH trajectories; excluding and filling frames;
    determining trajectory endpoints and interpolating."""
    def create_interp_traj(df_row, fine_dt, coarse_dt, common_ht_endpts):
        return ShoulderTrajInterp(df_row['Trial_Name'], df_row['ht'], df_row['gh'], df_row['st'],
                                  df_row['up_down_analysis'], fine_dt, coarse_dt, common_ht_endpts)

    db['ht'], db['gh'], db['st'] = zip(*db['Trial'].apply(get_trajs, args=[db.attrs['dt'], torso_def, use_ac]))
    if should_clean:
        clean_up_trials(db)
    if should_fill:
        fill_trials(db)
    db['up_down_analysis'] = db['ht'].apply(analyze_up_down)
    db['traj_interp'] = db.apply(create_interp_traj, axis=1, args=[dtheta_fine, dtheta_coarse, ht_endpts_common])


def clean_up_trials(db: pd.DataFrame) -> None:
    """Exclude frames from trials where subject has not gotten ready to perform thumb-up elevation yet."""
    # the subject is axially rotating their arm to prepare for a thumb up arm elevation in the first 47 frames for SA,
    # and first 69 frames for CA
    clean_up_tasks = {'N022_SA_t01': 47, 'N022_CA_t01': 69}
    for trial_name, start_frame in clean_up_tasks.items():
        ht = db.loc[trial_name, 'ht']
        gh = db.loc[trial_name, 'gh']
        st = db.loc[trial_name, 'st']
        db.loc[trial_name, 'ht'] = PoseTrajectory.from_ht(ht.ht[start_frame:, :, :], ht.dt, ht.frame_nums[start_frame:])
        db.loc[trial_name, 'gh'] = PoseTrajectory.from_ht(gh.ht[start_frame:, :, :], gh.dt, gh.frame_nums[start_frame:])
        db.loc[trial_name, 'st'] = PoseTrajectory.from_ht(st.ht[start_frame:, :, :], st.dt, st.frame_nums[start_frame:])


def fill_trials(db: pd.DataFrame) -> None:
    """Fill trials."""
    def create_quat(angle, axis):
        return q.from_float_array(np.concatenate((np.array([np.cos(angle/2)]), np.sin(angle/2) * axis)))

    def fill_traj(traj, frames_to_avg, frames_to_fill):
        dt = traj.dt

        # compute averages
        ang_vel_avg_up = np.mean(traj.ang_vel[0:frames_to_avg, :], axis=0)
        ang_vel_avg_up_angle = np.linalg.norm(ang_vel_avg_up)
        ang_vel_avg_up_axis = ang_vel_avg_up / ang_vel_avg_up_angle
        ang_vel_avg_down = np.mean(traj.ang_vel[-frames_to_avg:, :], axis=0)
        ang_vel_avg_down_angle = np.linalg.norm(ang_vel_avg_down)
        ang_vel_avg_down_axis = ang_vel_avg_down / ang_vel_avg_down_angle
        vel_avg_up = np.mean(traj.vel[0:frames_to_avg, :], axis=0)
        vel_avg_down = np.mean(traj.vel[-frames_to_avg:, :], axis=0)

        # add additional frames
        pos_up_filled = np.stack([vel_avg_up * dt * i + traj.pos[0] for i in range(-frames_to_fill, 0)], 0)
        pos_down_filled = np.stack([vel_avg_down * dt * i + traj.pos[-1] for i in range(1, frames_to_fill + 1)], 0)
        quat_up_filled = np.stack([create_quat(ang_vel_avg_up_angle * dt * i, ang_vel_avg_up_axis) * traj.quat[0]
                                  for i in range(-frames_to_fill, 0)], 0)
        quat_down_filled = np.stack([create_quat(ang_vel_avg_down_angle * dt * i, ang_vel_avg_down_axis) * traj.quat[-1]
                                    for i in range(1, frames_to_fill + 1)], 0)

        # create new trajectory
        new_frame_nums = np.concatenate((np.arange(traj.frame_nums[0] - frames_to_fill, traj.frame_nums[0]),
                                         traj.frame_nums,
                                         np.arange(traj.frame_nums[-1] + 1, traj.frame_nums[-1] + frames_to_fill + 1)))
        if new_frame_nums[0] < 0:
            new_frame_nums = new_frame_nums + (-new_frame_nums[0])

        pos = np.concatenate((pos_up_filled, traj.pos, pos_down_filled), axis=0)
        quat = q.as_float_array(np.concatenate((quat_up_filled, traj.quat, quat_down_filled), axis=0))

        return PoseTrajectory.from_quat(pos, quat, dt, new_frame_nums)

    # this trial is extremely close to reaching the 25 deg ht elevation mark both up (25.28) and down (26.17), so I have
    # elected to fill it because it will give us this datapoint for the rest of the trials
    db.loc['N003A_SA_t01', 'ht'] = fill_traj(db.loc['N003A_SA_t01', 'ht'], 5, 5)
    db.loc['N003A_SA_t01', 'gh'] = fill_traj(db.loc['N003A_SA_t01', 'gh'], 5, 5)
    db.loc['N003A_SA_t01', 'st'] = fill_traj(db.loc['N003A_SA_t01', 'st'], 5, 5)


def st_interp(traj_interp):
    st_joint = traj_interp.st
    st_joint.induced_axial_rot_up = traj_interp.get_up(st_joint.induced_axial_rot)
    st_joint.induced_axial_rot_down = traj_interp.get_down(st_joint.induced_axial_rot)
    st_joint.induced_axial_rot_sym_fine_up = \
        interp_vec_traj(traj_interp.ht_ea_up, st_joint.induced_axial_rot_up, traj_interp.sym_ht_range_fine)
    st_joint.induced_axial_rot_sym_fine_down = \
        interp_vec_traj(traj_interp.ht_ea_down, st_joint.induced_axial_rot_down, traj_interp.sym_ht_range_fine)
    st_joint.induced_axial_rot_common_fine_up = \
        interp_vec_traj(traj_interp.ht_ea_up, st_joint.induced_axial_rot_up, traj_interp.common_ht_range_fine)
    st_joint.induced_axial_rot_common_fine_down = \
        interp_vec_traj(traj_interp.ht_ea_down, st_joint.induced_axial_rot_down, traj_interp.common_ht_range_fine)
    st_joint.induced_axial_rot_common_coarse_up = \
        interp_vec_traj(traj_interp.ht_ea_up, st_joint.induced_axial_rot_up, traj_interp.common_ht_range_coarse)
    st_joint.induced_axial_rot_common_coarse_down = \
        interp_vec_traj(traj_interp.ht_ea_down, st_joint.induced_axial_rot_down, traj_interp.common_ht_range_coarse)


def st_induced_axial_rot_ang_vel(traj_interp):
    st_joint = traj_interp.st
    num_frames = st_joint.traj.rot_matrix.shape[0]
    # The ST velocity is expressed in the torso coordinate system so we can't just use its x,y,z components.
    # We need to first project it onto the x,y,z axes of the scapula. This is equivalent to expressing the ST velocity
    # in its own body coordinates.
    st_angvel_x = (extended_dot(st_joint.traj.ang_vel, st_joint.traj.rot_matrix[:, :, 0])[..., np.newaxis] *
                   st_joint.traj.rot_matrix[:, :, 0])
    st_angvel_y = (extended_dot(st_joint.traj.ang_vel, st_joint.traj.rot_matrix[:, :, 1])[..., np.newaxis] *
                   st_joint.traj.rot_matrix[:, :, 1])
    st_angvel_z = (extended_dot(st_joint.traj.ang_vel, st_joint.traj.rot_matrix[:, :, 2])[..., np.newaxis] *
                   st_joint.traj.rot_matrix[:, :, 2])

    st_angvel_proj_x = extended_dot(st_angvel_x, traj_interp.ht.traj.rot_matrix[:, :, 1])
    st_angvel_proj_y = extended_dot(st_angvel_y, traj_interp.ht.traj.rot_matrix[:, :, 1])
    st_angvel_proj_z = extended_dot(st_angvel_z, traj_interp.ht.traj.rot_matrix[:, :, 1])
    st_angvel_proj = extended_dot(st_joint.traj.ang_vel, traj_interp.ht.traj.rot_matrix[:, :, 1])

    st_joint.induced_axial_rot = np.empty((num_frames, 4), dtype=np.float)
    st_joint.induced_axial_rot[:, 0] = cumtrapz(st_angvel_proj_x, dx=st_joint.traj.dt, initial=0)
    st_joint.induced_axial_rot[:, 1] = cumtrapz(st_angvel_proj_y, dx=st_joint.traj.dt, initial=0)
    st_joint.induced_axial_rot[:, 2] = cumtrapz(st_angvel_proj_z, dx=st_joint.traj.dt, initial=0)
    st_joint.induced_axial_rot[:, 3] = cumtrapz(st_angvel_proj, dx=st_joint.traj.dt, initial=0)
    st_interp(traj_interp)


def st_induced_axial_rot_fha(traj_interp):
    st_joint = traj_interp.st
    num_frames = st_joint.traj.rot_matrix.shape[0]
    st_joint_diff = traj_interp.st.traj.quat[1:] * np.conjugate(traj_interp.st.traj.quat[:-1])
    st_joint_fha = q.as_rotation_vector(st_joint_diff)
    st_joint_fha_x = (extended_dot(st_joint_fha, traj_interp.st.traj.rot_matrix[:-1, :, 0])[..., np.newaxis] *
                      st_joint.traj.rot_matrix[:-1, :, 0])
    st_joint_fha_y = (extended_dot(st_joint_fha, traj_interp.st.traj.rot_matrix[:-1, :, 1])[..., np.newaxis] *
                      st_joint.traj.rot_matrix[:-1, :, 1])
    st_joint_fha_z = (extended_dot(st_joint_fha, traj_interp.st.traj.rot_matrix[:-1, :, 2])[..., np.newaxis] *
                      st_joint.traj.rot_matrix[:-1, :, 2])

    st_fha_proj_x = extended_dot(st_joint_fha_x, traj_interp.ht.traj.rot_matrix[:-1, :, 1])
    st_fha_proj_y = extended_dot(st_joint_fha_y, traj_interp.ht.traj.rot_matrix[:-1, :, 1])
    st_fha_proj_z = extended_dot(st_joint_fha_z, traj_interp.ht.traj.rot_matrix[:-1, :, 1])
    st_fha_proj = extended_dot(st_joint_fha, traj_interp.ht.traj.rot_matrix[:-1, :, 1])

    st_joint.induced_axial_rot = np.empty((num_frames, 4), dtype=np.float)
    st_joint.induced_axial_rot[0, :] = 0
    st_joint.induced_axial_rot[1:, 0] = np.add.accumulate(st_fha_proj_x)
    st_joint.induced_axial_rot[1:, 1] = np.add.accumulate(st_fha_proj_y)
    st_joint.induced_axial_rot[1:, 2] = np.add.accumulate(st_fha_proj_z)
    st_joint.induced_axial_rot[1:, 3] = np.add.accumulate(st_fha_proj)
    st_interp(traj_interp)


def quat_project(quat, axis):
    axis_project = np.dot(q.as_float_array(quat)[1:], axis) * axis
    quat_proj = np.quaternion(q.as_float_array(quat)[0], axis_project[0], axis_project[1], axis_project[2])
    return quat_proj / np.absolute(quat_proj)


def st_induced_axial_rot_swing_twist(traj_interp: ShoulderTrajInterp):
    st_joint = traj_interp.st
    num_frames = st_joint.traj.rot_matrix.shape[0]
    # rotational difference between frames expressed in torso coordinate system
    st_joint_diff = traj_interp.st.traj.quat[1:] * np.conjugate(traj_interp.st.traj.quat[:-1])

    induced_axial_rot_delta = np.empty((num_frames, 4), dtype=np.float)
    induced_axial_rot_delta[0, :] = 0
    for i in range(num_frames-1):
        hum_axis = traj_interp.ht.traj.rot_matrix[i, :, 1]

        # this computes the induced axial rotation from the all ST rotation
        rot_vec = q.as_rotation_vector(quat_project(st_joint_diff[i], hum_axis))
        rot_vec_theta = np.linalg.norm(rot_vec)
        rot_vec_axis = rot_vec / rot_vec_theta
        # Note that rot_vec_theta will always be + because of np.linalg.norm. But a rotation about an axis v by an angle
        # theta is the same as a rotation about -v by an angle -theta. So here the humeral axis sets our direction. That
        # is we always rotate around hum_axis (and not -hum_axis) and adjust the sign of rot_vec_theta accordingly
        induced_axial_rot_delta[i+1, 3] = rot_vec_theta * (1 if np.dot(rot_vec_axis, hum_axis) > 0 else -1)

        # this computes it for each individual axis of the scapula
        for j in range(3):
            # first project the scapula rotation onto one of its axis
            st_axis_proj = quat_project(st_joint_diff[i], traj_interp.st.traj.rot_matrix[i, :, j])
            # then proceed as above
            rot_vec = q.as_rotation_vector(quat_project(st_axis_proj, hum_axis))
            rot_vec_theta = np.linalg.norm(rot_vec)
            rot_vec_axis = rot_vec / rot_vec_theta
            induced_axial_rot_delta[i+1, j] = rot_vec_theta * (1 if np.dot(rot_vec_axis, hum_axis) > 0 else -1)

    st_joint.induced_axial_rot = np.add.accumulate(induced_axial_rot_delta)
    st_interp(traj_interp)
