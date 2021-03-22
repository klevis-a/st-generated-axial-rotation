"""Create PDF records for all elevation trials performing versus analysis such as up versus down, male vs female,
<35 versus >45, etc.

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

from typing import NamedTuple, Union
from functools import partial
import numpy as np
import quaternion as q
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.backends.backend_pdf import PdfPages
from biokinepy.trajectory import PoseTrajectory
from biokinepy.mean_smoother import quat_mean
from st_generated_axial_rot.common.python_utils import rgetattr
from st_generated_axial_rot.common.plot_utils import style_axes
from st_generated_axial_rot.common.analysis_utils import extract_sub_rot
import logging

log = logging.getLogger(__name__)


def traj_stats(all_traj):
    n = np.count_nonzero(~np.isnan(all_traj), axis=0)
    log.info('Available subjects: %s', n)
    traj_mean = np.nanmean(all_traj, axis=0)
    traj_std = np.nanstd(all_traj, axis=0)
    traj_se = traj_std/np.sqrt(n)
    return traj_mean, traj_std, traj_se


def quat_mean_trajs(trajs):
    mean_traj = np.empty((trajs.shape[1],), dtype=np.quaternion)
    for i in range(mean_traj.size):
        mean_traj[i] = q.from_float_array(quat_mean(q.as_float_array(trajs[:, i])))
    return mean_traj


def extract_sub_rot_diff(shoulder_traj, traj_def, y_def, decomp_method, sub_rot):
    # first extract ht, gh, or st
    joint_traj = getattr(shoulder_traj, traj_def)
    # Then extract the decomp_method. Note that each JointTrajectory actually computes separate scalar interpolation for
    # true_axial_rot (that's why I don't access the PoseTrajectory below) because true_axial_rot is path dependent so
    # it doesn't make sense to compute it on a trajectory that starts at 25 degrees (for example)
    if 'euler' in decomp_method:
        y_up = rgetattr(getattr(joint_traj, y_def + '_up'), decomp_method)[:, sub_rot]
        y_down = rgetattr(getattr(joint_traj, y_def + '_down'), decomp_method)[:, sub_rot]
    else:
        if sub_rot is None:
            y_up = getattr(joint_traj, decomp_method + '_' + y_def + '_up')
            y_down = getattr(joint_traj, decomp_method + '_' + y_def + '_down')
        else:
            y_up = getattr(joint_traj, decomp_method + '_' + y_def + '_up')[:, sub_rot]
            y_down = getattr(joint_traj, decomp_method + '_' + y_def + '_down')[:, sub_rot]

    return y_up - y_down


def extract_interp_quat_traj(shoulder_traj, traj_def, y_def):
    joint_traj = getattr(shoulder_traj, traj_def)
    return getattr(getattr(joint_traj, y_def), 'quat')


def ind_plotter(shoulder_traj, traj_def, x_def, y_def, decomp_method, sub_rot, ax):
    y = extract_sub_rot(shoulder_traj, traj_def, y_def, decomp_method, sub_rot)
    return ax.plot(getattr(shoulder_traj, x_def), np.rad2deg(y),
                   label='_'.join(shoulder_traj.trial_name.split('_')[0:2]), alpha=0.6)


def ind_diff_plotter(shoulder_traj, traj_def, x_def, y_def, decomp_method, sub_rot, ax):
    y_up = extract_sub_rot(shoulder_traj, traj_def, y_def + '_up', decomp_method, sub_rot)
    y_down = extract_sub_rot(shoulder_traj, traj_def, y_def + '_down', decomp_method, sub_rot)
    return ax.plot(getattr(shoulder_traj, x_def), np.rad2deg(y_up-y_down),
                   label='_'.join(shoulder_traj.trial_name.split('_')[0:2]), alpha=0.6)


def ind_plotter_color_spec(shoulder_traj, traj_def, x_def, y_def, decomp_method, sub_rot, ax, c):
    y = extract_sub_rot(shoulder_traj, traj_def, y_def, decomp_method, sub_rot)
    return ax.plot(getattr(shoulder_traj, x_def), np.rad2deg(y), color=c, alpha=0.6)


def summary_plotter(shoulder_trajs, traj_def, x_ind_def, y_ind_def, x_cmn_def, y_cmn_def, decomp_method, sub_rot, ax,
                    ind_plotter_fnc, avg_color, quat_avg_color):
    # plot individual trajectories
    ind_traj_plot_lines = shoulder_trajs.apply(ind_plotter_fnc,
                                               args=[traj_def, x_ind_def, y_ind_def, decomp_method, sub_rot, ax])
    # common x-axis : only look at the first trajectory because it will be the same for all
    x_cmn = getattr(shoulder_trajs.iloc[0], x_cmn_def)
    # Extract the y-values spanning the common x-axis. This goes to each quaternion interpolated trajectory, and applies
    # decomp_method and sub_rot. The results are then averaged below. This is technically not correct because
    # mathematically it doesn't make sense to average PoE, Elevation, etc. but this is how other papers handle this step
    y_cmn_all = np.stack(shoulder_trajs.apply(extract_sub_rot, args=[traj_def, y_cmn_def, decomp_method, sub_rot]), 0)
    traj_mean, traj_std, traj_se = traj_stats(y_cmn_all)
    agg_lines = ax.errorbar(x_cmn, np.rad2deg(traj_mean), yerr=np.rad2deg(traj_se), capsize=2, color=avg_color,
                            zorder=4, lw=2)
    # So here the individual interpolated trajectories are averaged via quaternions. This is mathematically correct.
    # Then the averaged trajectory is decomposed according to decomp_method and sub_rot. One could then use this to
    # compute SD and SE, but I don't go that far since almost always the quaternion mean and the mean as computed above
    # match very well. But I do overlay the quaternion mean as a sanity check
    if 'euler' in decomp_method:
        mean_traj_quat = quat_mean_trajs(np.stack(shoulder_trajs.apply(extract_interp_quat_traj,
                                                                       args=[traj_def, y_cmn_def]), axis=0))
        mean_traj_pos = np.zeros((mean_traj_quat.size, mean_traj_quat.size))
        mean_traj_pose = PoseTrajectory.from_quat(mean_traj_pos, q.as_float_array(mean_traj_quat))
        mean_y_quat = rgetattr(mean_traj_pose, decomp_method)[:, sub_rot]
        quat_mean_lines = ax.plot(x_cmn, np.rad2deg(mean_y_quat), color=quat_avg_color, zorder=5, lw=2)
    else:
        quat_mean_lines = None

    return ind_traj_plot_lines, agg_lines, quat_mean_lines


def summary_diff_plotter(shoulder_trajs, traj_def, x_ind_def, y_ind_def, x_cmn_def, y_cm_def, decomp_method, sub_rot,
                         ax, avg_color):
    # plot individual trajectories
    ind_traj_plot_lines = shoulder_trajs.apply(ind_diff_plotter,
                                               args=[traj_def, x_ind_def, y_ind_def, decomp_method, sub_rot, ax])
    # common x-axis : only look at the first trajectory because it will be the same for all
    x_cmn = getattr(shoulder_trajs.iloc[0], x_cmn_def)
    y_cmn_all = np.stack(shoulder_trajs.apply(extract_sub_rot_diff,
                                              args=[traj_def, y_cm_def, decomp_method, sub_rot]), 0)
    traj_mean, traj_std, traj_se = traj_stats(y_cmn_all)
    agg_lines = ax.errorbar(x_cmn, np.rad2deg(traj_mean), yerr=np.rad2deg(traj_se), capsize=2, color=avg_color,
                            zorder=4, lw=2)
    return ind_traj_plot_lines, agg_lines


def grp_plotter(activity_df, col_name, pdf_file_path, grp_c, plot_directives, section):
    with PdfPages(pdf_file_path) as grp_pdf:
        for name, plot_dir in plot_directives.items():
            fig = plt.figure()
            ax = fig.subplots()
            ind_lines = []
            ind_leg_entries = []
            avg_lines = []
            avg_entries = []
            quat_avg_lines = []
            quat_avg_entries = []
            for grp, grp_df in activity_df.groupby(col_name):
                ind_plotter_fnc = partial(ind_plotter_color_spec, c=grp_c[grp][0])
                ind_traj_plot_lines, agg_lines, quat_mean_lines = summary_plotter(
                    grp_df['traj_interp'], plot_dir.traj, section[0], section[1], section[2], section[3],
                    plot_dir.decomp_method, plot_dir.sub_rot, ax, ind_plotter_fnc, grp_c[grp][1],
                    grp_c[grp][2])
                ind_lines.append(ind_traj_plot_lines.iloc[0][0])
                ind_leg_entries.append(grp)
                avg_lines.append(agg_lines[0])
                avg_entries.append(grp + r' Mean$\pm$SE')
                if quat_mean_lines is not None:
                    quat_avg_lines.append(quat_mean_lines[0])
                    quat_avg_entries.append(grp + ' Mean(Quat)')
            ax.legend(ind_lines, ind_leg_entries, handlelength=0.75, handletextpad=0.25, columnspacing=0.5,
                      loc='lower right')
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            style_axes(ax, 'Humerothoracic Elevation (Deg)', plot_dir.y_label)
            fig.tight_layout()
            fig.suptitle(plot_dir.title)
            if quat_avg_lines:
                fig.legend(avg_lines + quat_avg_lines, avg_entries + quat_avg_entries, loc='lower left',
                           handlelength=0.75, handletextpad=0.25, ncol=2)
            else:
                fig.legend(avg_lines, avg_entries, loc='lower left', handlelength=0.75, handletextpad=0.25)
            grp_pdf.savefig(fig)
            fig.clf()
            plt.close(fig)


class PlotDirective(NamedTuple):
    """Class containing directives for what to plot.

    Attributes
    ----------
    traj: str
        Whether to plot HT, ST, or GH.
    sub_rot: int, None
        Which sub-rotation to plot (0, 1, 2), or None for true_axial_rot.
    decomp_method: str
        Which decomposition to perform within PoseTrajectory: e.g. true_axial_rot, euler.ht_isb, euler.gh_isb,
        euler.ht_phadke, etc.
    y_label: str
        Y label for the plot.
    title: str
        Title string for the plot.
    """
    traj: str
    sub_rot: Union[int, None]
    decomp_method: str
    y_label: str
    title: str


plot_directives = {
    'gh_poe_isb': PlotDirective('gh', 0, 'euler.gh_isb', 'Plane of elevation (deg)', 'GH Plane of Elevation ISB'),
    'gh_ea_isb': PlotDirective('gh', 1, 'euler.gh_isb', 'Elevation (deg)', 'GH Elevation ISB'),
    'gh_axial_isb': PlotDirective('gh', 2, 'euler.gh_isb', 'Axial Rotation (deg)', 'GH Axial Rotation ISB'),
    'gh_true_axial': PlotDirective('gh', None, 'true_axial_rot', 'Axial Rotation (deg)', 'GH True Axial Rotation'),
    'ht_poe_isb': PlotDirective('ht', 0, 'euler.ht_isb', 'Plane of elevation (deg)', 'HT Plane of Elevation ISB'),
    'ht_ea_isb': PlotDirective('ht', 1, 'euler.ht_isb', 'Elevation (deg)', 'HT Elevation ISB'),
    'ht_axial_isb': PlotDirective('ht', 2, 'euler.ht_isb', 'Axial Rotation (deg)', 'HT Axial Rotation ISB'),
    'ht_true_axial': PlotDirective('ht', None, 'true_axial_rot', 'Axial Rotation (deg)', 'HT True Axial Rotation'),
    'gh_poe_phadke': PlotDirective('gh', 1, 'euler.gh_phadke', 'Plane of elevation (deg)', 'GH Horz Abd/Flex Phadke'),
    'gh_ea_phadke': PlotDirective('gh', 0, 'euler.gh_phadke', 'Elevation (deg)', 'GH Elevation Phadke'),
    'gh_axial_phadke': PlotDirective('gh', 2, 'euler.gh_phadke', 'Axial Rotation (deg)', 'GH Axial Rotation Phadke'),
    'ht_poe_phadke': PlotDirective('ht', 1, 'euler.ht_phadke', 'Plane of elevation (deg)', 'HT Horz Abd/Flex Phadke'),
    'ht_ea_phadke': PlotDirective('ht', 0, 'euler.ht_phadke', 'Elevation (deg)', 'HT Elevation Phadke'),
    'ht_axial_phadke': PlotDirective('ht', 2, 'euler.ht_phadke', 'Axial Rotation (deg)', 'HT Axial Rotation Phadke'),
    'st_protraction_isb': PlotDirective('st', 0, 'euler.st_isb', 'Pro/Retraction (deg)', 'ST Pro/Retraction'),
    'st_latmed_isb': PlotDirective('st', 1, 'euler.st_isb', 'Lateral/Medial (deg)', 'ST Lateral/Medial'),
    'st_tilt_isb': PlotDirective('st', 2, 'euler.st_isb', 'Tilt (deg)', 'ST Tilt'),
    'st_induced_axial_x': PlotDirective('st', 0, 'induced_axial_rot', 'Axial Rotation (deg)',
                                        'ST Induced Axial Rotation Lat/Med'),
    'st_induced_axial_y': PlotDirective('st', 1, 'induced_axial_rot', 'Axial Rotation (deg)',
                                        'ST Induced Axial Rotation Re/Pro'),
    'st_induced_axial_z': PlotDirective('st', 2, 'induced_axial_rot', 'Axial Rotation (deg)',
                                        'ST Induced Axial Rotation Tilt')}

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, add_st_induced, st_induced_axial_rot_fha
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Overview of elevations trials', __package__, __file__))
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

    if bool(distutils.util.strtobool(params.weighted)):
        db_elev = db.loc[db['Trial_Name'].str.contains('_WCA_|_WSA_|_WFE_')].copy()
    else:
        db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()

    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_elev['traj_interp'].apply(add_st_induced, args=[st_induced_axial_rot_fha])

    sections = [('ht_ea_up', 'up', 'common_ht_range_coarse', 'common_coarse_up', 'up'),
                ('ht_ea_down', 'down', 'common_ht_range_coarse', 'common_coarse_down', 'down')]

    # we need more colors to encompass all subjects than the default color scalar offers
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20c.colors)
    plot_utils.init_graphing(params.backend)
    for activity, activity_df in db_elev.groupby('Activity', observed=True):
        # plot up and down separately
        for section in sections:
            pdf_file_path = output_path / (activity + '_' + params.torso_def + ('_' + params.scap_lateral) + '_' +
                                           section[4] + '.pdf')
            with PdfPages(pdf_file_path) as activity_pdf:
                for name, plot_dir in plot_directives.items():
                    fig = plt.figure()
                    ax = fig.subplots()
                    _, agg_lines, quat_mean_lines = summary_plotter(
                        activity_df['traj_interp'], plot_dir.traj, section[0], section[1], section[2], section[3],
                        plot_dir.decomp_method, plot_dir.sub_rot, ax, ind_plotter, 'black', 'red')
                    if quat_mean_lines:
                        ax.legend([agg_lines.lines[0], quat_mean_lines[0]], [r'Mean$\pm$SE', 'Mean(Quat)'],
                                  loc='lower right', handlelength=0.75, handletextpad=0.25)
                    else:
                        ax.legend([agg_lines.lines[0]], [r'Mean$\pm$SE'], loc='lower right', handlelength=0.75,
                                  handletextpad=0.25)
                    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                    style_axes(ax, 'Humerothoracic Elevation (Deg)', plot_dir.y_label)
                    fig.tight_layout()
                    fig.subplots_adjust(bottom=0.2)
                    fig.suptitle(plot_dir.title)
                    fig.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                               fontsize=8)
                    activity_pdf.savefig(fig)
                    fig.clf()
                    plt.close(fig)

            # plot by gender
            pdf_file_path = output_path / (activity + '_' + params.torso_def + ('_' + params.scap_lateral) + '_' +
                                           section[4] + '_gender.pdf')
            gender_c = {'F': ['lightcoral', 'red', 'darkorange'], 'M': ['dodgerblue', 'blue', 'navy']}
            grp_plotter(activity_df, 'Gender', pdf_file_path, gender_c, plot_directives, section)

            # plot by age
            pdf_file_path = output_path / (activity + '_' + params.torso_def + ('_' + params.scap_lateral) + '_' +
                                           section[4] + '_age.pdf')
            age_c = {'>45': ['lightcoral', 'red', 'darkorange'], '<35': ['dodgerblue', 'blue', 'navy']}
            grp_plotter(activity_df, 'age_group', pdf_file_path, age_c, plot_directives, section)

        # plot up down difference
        pdf_file_path = output_path / (activity + '_' + params.torso_def + ('_' + params.scap_lateral) + '_diff.pdf')
        with PdfPages(pdf_file_path) as diff_pdf:
            for name, plot_dir in plot_directives.items():
                fig = plt.figure()
                ax = fig.subplots()
                _, agg_lines = summary_diff_plotter(
                    activity_df['traj_interp'], plot_dir.traj, 'sym_ht_range_fine', 'sym_fine',
                    'common_ht_range_coarse', 'common_coarse', plot_dir.decomp_method, plot_dir.sub_rot, ax, 'black')
                ax.legend([agg_lines.lines[0]], [r'Mean$\pm$SE'], loc='lower right', handlelength=0.75,
                          handletextpad=0.25)
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                style_axes(ax, 'Humerothoracic Elevation (Deg)', plot_dir.y_label)
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.2)
                fig.suptitle(plot_dir.title)
                fig.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                           fontsize=8)
                diff_pdf.savefig(fig)
                fig.clf()
                plt.close(fig)
