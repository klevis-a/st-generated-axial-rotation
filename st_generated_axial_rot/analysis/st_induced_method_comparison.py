"""Compare different methods of computing ST induced HT axial rotation

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
use_ac: Whether to use the AC or GC landmark when building the scapula CS.
"""
import numpy as np
import quaternion as q
from scipy.integrate import cumtrapz
from biokinepy.vec_ops import extended_dot
from st_generated_axial_rot.common.analysis_utils import quat_project


def st_induced_axial_rot_ang_vel(st, ht):
    num_frames = st.rot_matrix.shape[0]
    # The ST velocity is expressed in the torso coordinate system so we can't just use its x,y,z components.
    # We need to first project it onto the x,y,z axes of the scapula. This is equivalent to expressing the ST velocity
    # in its own body coordinates.
    st_angvel_x = (extended_dot(st.ang_vel, st.rot_matrix[:, :, 0])[..., np.newaxis] * st.rot_matrix[:, :, 0])
    st_angvel_y = (extended_dot(st.ang_vel, st.rot_matrix[:, :, 1])[..., np.newaxis] * st.rot_matrix[:, :, 1])
    st_angvel_z = (extended_dot(st.ang_vel, st.rot_matrix[:, :, 2])[..., np.newaxis] * st.rot_matrix[:, :, 2])

    st_angvel_proj_x = extended_dot(st_angvel_x, ht.rot_matrix[:, :, 1])
    st_angvel_proj_y = extended_dot(st_angvel_y, ht.rot_matrix[:, :, 1])
    st_angvel_proj_z = extended_dot(st_angvel_z, ht.rot_matrix[:, :, 1])
    st_angvel_proj = extended_dot(st.ang_vel, ht.rot_matrix[:, :, 1])

    induced_axial_rot = np.empty((num_frames, 4), dtype=np.float)
    induced_axial_rot[:, 0] = cumtrapz(st_angvel_proj_x, dx=st.dt, initial=0)
    induced_axial_rot[:, 1] = cumtrapz(st_angvel_proj_y, dx=st.dt, initial=0)
    induced_axial_rot[:, 2] = cumtrapz(st_angvel_proj_z, dx=st.dt, initial=0)
    induced_axial_rot[:, 3] = cumtrapz(st_angvel_proj, dx=st.dt, initial=0)

    return induced_axial_rot


def st_induced_axial_rot_fha(st, ht):
    num_frames = st.rot_matrix.shape[0]
    st_diff = st.quat[1:] * np.conjugate(st.quat[:-1])
    st_fha = q.as_rotation_vector(st_diff)
    st_fha_x = (extended_dot(st_fha, st.rot_matrix[:-1, :, 0])[..., np.newaxis] * st.rot_matrix[:-1, :, 0])
    st_fha_y = (extended_dot(st_fha, st.rot_matrix[:-1, :, 1])[..., np.newaxis] * st.rot_matrix[:-1, :, 1])
    st_fha_z = (extended_dot(st_fha, st.rot_matrix[:-1, :, 2])[..., np.newaxis] * st.rot_matrix[:-1, :, 2])

    st_fha_proj_x = extended_dot(st_fha_x, ht.rot_matrix[:-1, :, 1])
    st_fha_proj_y = extended_dot(st_fha_y, ht.rot_matrix[:-1, :, 1])
    st_fha_proj_z = extended_dot(st_fha_z, ht.rot_matrix[:-1, :, 1])
    st_fha_proj = extended_dot(st_fha, ht.rot_matrix[:-1, :, 1])

    induced_axial_rot = np.empty((num_frames, 4), dtype=np.float)
    induced_axial_rot[0, :] = 0
    induced_axial_rot[1:, 0] = np.add.accumulate(st_fha_proj_x)
    induced_axial_rot[1:, 1] = np.add.accumulate(st_fha_proj_y)
    induced_axial_rot[1:, 2] = np.add.accumulate(st_fha_proj_z)
    induced_axial_rot[1:, 3] = np.add.accumulate(st_fha_proj)

    return induced_axial_rot


def st_induced_axial_rot_swing_twist(st, ht):
    num_frames = st.rot_matrix.shape[0]
    # rotational difference between frames expressed in torso coordinate system
    st_diff = st.quat[1:] * np.conjugate(st.quat[:-1])

    induced_axial_rot_delta = np.empty((num_frames, 4), dtype=np.float)
    induced_axial_rot_delta[0, :] = 0
    for i in range(num_frames-1):
        hum_axis = ht.rot_matrix[i, :, 1]

        # this computes the induced axial rotation from the ST rotation in its entirety
        rot_vec = q.as_rotation_vector(quat_project(st_diff[i], hum_axis))
        rot_vec_theta = np.linalg.norm(rot_vec)
        rot_vec_axis = rot_vec / rot_vec_theta
        # Note that rot_vec_theta will always be + because of np.linalg.norm. But a rotation about an axis v by an angle
        # theta is the same as a rotation about -v by an angle -theta. So here the humeral axis sets our direction. That
        # is, we always rotate around hum_axis (and not -hum_axis) and adjust the sign of rot_vec_theta accordingly
        induced_axial_rot_delta[i+1, 3] = rot_vec_theta * (1 if np.dot(rot_vec_axis, hum_axis) > 0 else -1)

        # this computes it for each individual axis of the scapula
        for j in range(3):
            # first project the scapula rotation onto one of its axis
            st_axis_proj = quat_project(st_diff[i], st.rot_matrix[i, :, j])
            # then proceed as above
            rot_vec = q.as_rotation_vector(quat_project(st_axis_proj, hum_axis))
            rot_vec_theta = np.linalg.norm(rot_vec)
            rot_vec_axis = rot_vec / rot_vec_theta
            induced_axial_rot_delta[i+1, j] = rot_vec_theta * (1 if np.dot(rot_vec_axis, hum_axis) > 0 else -1)

    return np.add.accumulate(induced_axial_rot_delta)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare different methods of computing ST induced HT axial rotation',
                                     __package__, __file__))
    params = get_params(config_dir / 'parameters.json')

    if not bool(distutils.util.strtobool(os.getenv('VARS_RETAINED', 'False'))):
        # ready db
        db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject, include_anthro=True)
        db['age_group'] = db['Age'].map(lambda age: '<35' if age < 40 else '>45')
        if params.excluded_trials:
            db = db[~db['Trial_Name'].str.contains('|'.join(params.excluded_trials))]
        db['Trial'].apply(pre_fetch)

    # relevant parameters
    output_path = Path(params.output_dir)
    use_ac = bool(distutils.util.strtobool(params.use_ac))

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # prepare db
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    db_rot = db.loc[db['Trial_Name'].str.contains('_ERa90_|_ERaR_')].copy()
    prepare_db(db_elev, params.torso_def, use_ac, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_rot = ready_er_db(db_rot, params.torso_def, use_ac, params.erar_endpts, params.era90_endpts,
                         params.dtheta_fine)

    # compute st-induced axial rotation with each of the three different methods
    db_elev['st_induced_ang_vel'] = db_elev[['st', 'ht']].apply(lambda x: st_induced_axial_rot_ang_vel(*x), axis=1)
    db_elev['st_induced_fha'] = db_elev[['st', 'ht']].apply(lambda x: st_induced_axial_rot_fha(*x), axis=1)
    db_elev['st_induced_swing_twist'] = db_elev[['st', 'ht']].apply(lambda x: st_induced_axial_rot_swing_twist(*x),
                                                                    axis=1)
    db_rot['st_induced_ang_vel'] = db_rot[['st', 'ht']].apply(lambda x: st_induced_axial_rot_ang_vel(*x), axis=1)
    db_rot['st_induced_fha'] = db_rot[['st', 'ht']].apply(lambda x: st_induced_axial_rot_fha(*x), axis=1)
    db_rot['st_induced_swing_twist'] = db_rot[['st', 'ht']].apply(lambda x: st_induced_axial_rot_swing_twist(*x),
                                                                  axis=1)

    # compare differences
    db_elev['ang_vel_vs_fha'] = db_elev['st_induced_ang_vel'] - db_elev['st_induced_fha']
    db_elev['ang_vel_vs_swing_twist'] = db_elev['st_induced_ang_vel'] - db_elev['st_induced_swing_twist']
    db_elev['fha_vs_swing_twist'] = db_elev['st_induced_fha'] - db_elev['st_induced_swing_twist']
    db_rot['ang_vel_vs_fha'] = db_rot['st_induced_ang_vel'] - db_rot['st_induced_fha']
    db_rot['ang_vel_vs_swing_twist'] = db_rot['st_induced_ang_vel'] - db_rot['st_induced_swing_twist']
    db_rot['fha_vs_swing_twist'] = db_rot['st_induced_fha'] - db_rot['st_induced_swing_twist']

    db_elev['ang_vel_vs_fha_max'] = \
        db_elev['ang_vel_vs_fha'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_elev['ang_vel_vs_swing_twist_max'] = \
        db_elev['ang_vel_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_elev['fha_vs_swing_twist_max'] = \
        db_elev['fha_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['ang_vel_vs_fha_max'] = \
        db_rot['ang_vel_vs_fha'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['ang_vel_vs_swing_twist_max'] = \
        db_rot['ang_vel_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['fha_vs_swing_twist_max'] = \
        db_rot['fha_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)

    print('ELEVATION')
    print('Angular Velocity vs FHA Maximum Difference: {:.2f}'.format(db_elev['ang_vel_vs_fha_max'].max()))
    print('Angular Velocity vs Swing Twist Maximum Difference: {:.2f}'.format(db_elev['ang_vel_vs_swing_twist_max']
                                                                              .max()))
    print('FHA vs Swing Twist Maximum Difference: {:.2f}'.format(db_elev['fha_vs_swing_twist_max'].max()))

    print('INTERNAL/EXTERNAL ROTATION')
    print('Angular Velocity vs FHA Maximum Difference: {:.2f}'.format(db_rot['ang_vel_vs_fha_max'].max()))
    print('Angular Velocity vs Swing Twist Maximum Difference: {:.2f}'.format(db_rot['ang_vel_vs_swing_twist_max']
                                                                              .max()))
    print('FHA vs Swing Twist Maximum Difference: {:.2f}'.format(db_rot['fha_vs_swing_twist_max'].max()))
