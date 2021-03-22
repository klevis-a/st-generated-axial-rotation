"""Compute remaining HT rotation after elevation has been removed for individual trials

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
dtheta_coarse: Incremental angle (deg) to use for coarse interpolation between minimum and maximum HT elevation analyzed.
min_elev: Minimum HT elevation angle (deg) utilized for analysis that encompasses all trials.
max_elev: Maximum HT elevation angle (deg) utilized for analysis that encompasses all trials.
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
output_dir: Directory where PDF records should be output.
"""

from scipy.spatial.transform import Rotation
import quaternion as q
from biokinepy.trajectory import PoseTrajectory
from biokinepy.vec_ops import extended_dot
from st_generated_axial_rot.common.analysis_utils_contrib import compute_elev_axis

# these two methods are equivalent to each other - keeping them both here for reference

# def st_gh_balance(ht_traj: PoseTrajectory, st_traj: PoseTrajectory, gh_traj: PoseTrajectory):
#     num_frames = st_traj.rot_matrix.shape[0]
#     elev_axes = compute_elev_axis(ht_traj.rot_matrix)
#     hum_axes = ht_traj.rot_matrix[:, :, 1]
#     elev_axes_scap = np.squeeze(q.as_rotation_matrix(np.conjugate(st_traj.quat)) @ elev_axes[..., np.newaxis])
#
#     st_diff = st_traj.quat[1:] * np.conjugate(st_traj.quat[:-1])
#     gh_diff = gh_traj.quat[1:] * np.conjugate(gh_traj.quat[:-1])
#
#     st_proj = np.empty_like(st_diff)
#     gh_proj = np.empty_like(gh_diff)
#
#     for i in range(num_frames-1):
#         st_proj[i] = quat_project(st_diff[i], elev_axes[i])
#         gh_proj[i] = quat_project(gh_diff[i], elev_axes_scap[i])
#
#     st_remain = st_diff * np.conjugate(st_proj)
#     gh_remain = gh_diff * np.conjugate(gh_proj)
#
#     # use tensor transformation law to bring gh into torso
#     comb = st_remain * (st_traj.quat[:-1] * gh_remain * np.conjugate(st_traj.quat[:-1]))
#     comb_float = q.as_float_array(comb)
#     comb_float = np.concatenate((comb_float[:, 1:], comb_float[:, 0][..., np.newaxis]), 1)
#     comb_vec = Rotation.from_quat(comb_float).as_rotvec()
#
#     comb_angle = np.sqrt(extended_dot(comb_vec, comb_vec))
#     comb_axis = comb_vec / comb_angle[..., np.newaxis]
#     comb_angle = np.where(extended_dot(comb_axis, hum_axes[:-1]) > 0, comb_angle, -comb_angle)
#
#     return np.cumsum(comb_angle)


def st_gh_balance(ht_traj: PoseTrajectory, st_traj: PoseTrajectory, gh_traj: PoseTrajectory):
    num_frames = st_traj.rot_matrix.shape[0]
    elev_axes = compute_elev_axis(ht_traj.rot_matrix)
    hum_axes = ht_traj.rot_matrix[:, :, 1]

    ht_diff = ht_traj.quat[1:] * np.conjugate(ht_traj.quat[:-1])
    ht_proj = np.empty_like(ht_diff)

    for i in range(num_frames-1):
        ht_proj[i] = quat_project(ht_diff[i], elev_axes[i])

    ht_remain = ht_diff * np.conjugate(ht_proj)
    ht_remain_float = q.as_float_array(ht_remain)
    ht_remain_float = np.concatenate((ht_remain_float[:, 1:], ht_remain_float[:, 0][..., np.newaxis]), 1)
    ht_remain_vec = Rotation.from_quat(ht_remain_float).as_rotvec()

    ht_remain_angle = np.sqrt(extended_dot(ht_remain_vec, ht_remain_vec))
    ht_remain_axis = ht_remain_vec / ht_remain_angle[..., np.newaxis]
    ht_remain_angle = np.where(extended_dot(ht_remain_axis, hum_axes[:-1]) > 0, ht_remain_angle, -ht_remain_angle)

    return np.cumsum(ht_remain_angle)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, quat_project
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compute remaining HT rotation after elevation has been removed '
                                     'for individual trials', __package__, __file__))
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

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # prepare db
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])

    db_elev['comb_angle'] = db_elev.apply(lambda df_row: st_gh_balance(df_row['ht'], df_row['st'], df_row['gh']),
                                          axis=1)

#%%
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20c.colors)
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    pdf_file_path = output_path / ('contrib_remaining_' + params.torso_def + ('_' + params.scap_lateral) + '.pdf')
    with PdfPages(pdf_file_path) as output_pdf:
        for activity, activity_df in db_elev.groupby('Activity', observed=True):
            fig = plt.figure()
            ax = fig.subplots()
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            plot_utils.style_axes(ax, 'HT Elevation Angle', 'Rotation (deg)')

            for trial_name, df_row in activity_df.iterrows():
                ht_traj = df_row['ht']
                st_traj = df_row['st']
                gh_traj = df_row['gh']
                up_down = df_row['up_down_analysis']

                start_idx = up_down.max_run_up_start_idx
                end_idx = up_down.max_run_up_end_idx
                sec = slice(start_idx,  end_idx + 1)

                ht_elev = np.rad2deg(-ht_traj.euler.ht_isb[sec, 1])
                ax.plot(ht_elev, np.rad2deg(df_row['comb_angle'][sec]), label='_'.join(trial_name.split('_')[0:2]))

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.2)
            fig.suptitle(activity)
            fig.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                       fontsize=8)
            output_pdf.savefig(fig)
            fig.clf()
            plt.close(fig)
