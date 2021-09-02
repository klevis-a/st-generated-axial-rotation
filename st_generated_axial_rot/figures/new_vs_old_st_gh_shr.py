"""Compare old SHR (GH Elev/ST UR) and new SHR based on contribution of GH and ST joint to HT elevation

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
dpi: Dots (pixels) per inch for generated figure. (e.g. 300)
fig_file: Path to file where to save figure.
"""
import numpy as np
from biokinepy.np_utils import find_runs


def shr_compute(df_row):
    st_contrib = df_row['traj_interp'].st.contribs[:, 1]
    gh_contrib = df_row['traj_interp'].gh.contribs[:, 1]
    ht_traj = df_row['ht']
    st_traj = df_row['st']
    gh_traj = df_row['gh']
    up_down = df_row['up_down_analysis']

    start_idx = up_down.max_run_up_start_idx
    end_idx = up_down.max_run_up_end_idx
    sec = slice(start_idx, end_idx + 1)

    st_elev = -np.rad2deg(st_contrib[sec] - st_contrib[start_idx])
    run_vals_new, run_starts_new, run_lengths_new = find_runs(st_elev > 1)
    first_gt_1_new = run_starts_new[np.nonzero(np.logical_and(run_lengths_new > 10, run_vals_new))[0][0]]

    st_elev_diff = np.rad2deg(-(st_traj.euler.st_isb[sec, 1] - st_traj.euler.st_isb[start_idx, 1]))
    run_vals_old, run_starts_old, run_lengths_old = find_runs(st_elev_diff > 1)
    first_gt_1_old = run_starts_old[np.nonzero(np.logical_and(run_lengths_old > 10, run_vals_old))[0][0]]

    start_idx_comp = max(first_gt_1_new, first_gt_1_old)
    ht_elev_diff = np.rad2deg(-(ht_traj.euler.ht_isb[sec, 1] - ht_traj.euler.ht_isb[start_idx, 1]))
    gh_elev_diff = np.rad2deg(-(gh_traj.euler.gh_isb[sec, 1] - gh_traj.euler.gh_isb[start_idx, 1]))

    ht_elev_range = ht_elev_diff[start_idx_comp:] - ht_elev_diff[start_idx_comp]
    old_shr = gh_elev_diff[start_idx_comp:] / st_elev_diff[start_idx_comp:]
    new_shr = ((gh_contrib[sec] - gh_contrib[start_idx])[start_idx_comp:] /
               (st_contrib[sec] - st_contrib[start_idx])[start_idx_comp:])
    ht_interp_range = np.arange(0, 100 + 0.25, 0.25)
    old_shr_interp = np.interp(ht_interp_range, ht_elev_range, old_shr, left=np.nan, right=np.nan)
    new_shr_interp = np.interp(ht_interp_range, ht_elev_range, new_shr, left=np.nan, right=np.nan)

    return old_shr, new_shr, old_shr_interp, new_shr_interp


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import matplotlib.pyplot as plt
    import spm1d
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.plot_utils import style_axes, mean_sd_plot, make_interactive, sig_filter, \
    extract_sig, output_spm_p
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare old SHR (GH Elev/ST UR) and new SHR based on contribution of GH and ST '
                                     'joint to HT elevation',__package__, __file__))
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
    db_elev['traj_interp'].apply(add_st_gh_contrib)
    db_elev['old_shr'], db_elev['new_shr'], db_elev['old_shr_interp'], db_elev['new_shr_interp'] = \
        zip(*db_elev.apply(shr_compute, axis=1))

#%%
    if bool(distutils.util.strtobool(params.parametric)):
        spm_test = spm1d.stats.ttest_paired
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest_paired
        infer_params = {'force_iterations': True}

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    plot_utils.init_graphing(params.backend)
    plt.close('all')

    x = np.arange(0, 100 + 0.25, 0.25)
    fig = plt.figure(figsize=(110 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(3, 1)

    for i in range(3):
        style_axes(axs[i], 'Motion Completion (%)' if i == 2 else None, 'SHR')
        axs[i].set_ylim(0, 12)

    leg_mean = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        old_shr = np.stack(activity_df['old_shr_interp'], axis=0)
        new_shr = np.stack(activity_df['new_shr_interp'], axis=0)

        # means and standard deviations
        old_shr_mean = np.mean(old_shr, axis=0)
        new_shr_mean = np.mean(new_shr, axis=0)

        old_shr_sd = np.std(old_shr, ddof=1, axis=0)
        new_shr_sd = np.std(new_shr, ddof=1, axis=0)

        # spm
        spm_res = spm_test(old_shr, new_shr).inference(alpha, two_tailed=True, **infer_params)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        old_shr_ln = mean_sd_plot(axs[cur_row], x, old_shr_mean, old_shr_sd,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        new_shr_ln = mean_sd_plot(axs[cur_row], x, new_shr_mean, new_shr_sd,
                                  dict(color=color_map.colors[1], alpha=0.25),
                                  dict(color=color_map.colors[1], marker=markers[0], markevery=20))

        # plot spm
        x_sig = sig_filter(spm_res, x)
        axs[cur_row].plot(x_sig, np.repeat(12, x_sig.size), color='k', lw=2)

        # print out
        print(activity)
        print('New SHR at 0: {:.2f}'.format(new_shr_mean[0]))
        print('New SHR at 100: {:.2f}'.format(new_shr_mean[-1]))
        print('Old SHR at 0: {:.2f}'.format(old_shr_mean[0]))
        print('Old SHR at 100: {:.2f}'.format(old_shr_mean[-1]))
        print('Motion % significant')
        print(extract_sig(spm_res, x))
        print(output_spm_p(spm_res))

        if idx == 0:
            leg_mean.append(old_shr_ln[0])
            leg_mean.append(new_shr_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('SHR Comparison', x=0.47, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend(leg_mean[:2], ['Euler SHR', 'Coordinated SHR'], loc='upper right', bbox_to_anchor=(1, 0.92),
                          ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, labelspacing=0.3,
                          borderpad=0.2)

    # add axes titles
    _, y0, _, h = axs[0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
