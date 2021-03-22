"""Compare the difference between HT elevation and (GH elevation + ST upward rotation) and
(GH + ST contributions to elevation)

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

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import spm1d
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.plot_utils import (style_axes, mean_sd_plot, make_interactive, sig_filter,
                                                          extract_sig, output_spm_p)
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare the difference between HT elevation and '
                                     '(GH elevation + ST upward rotation) and (GH + ST contributions to elevation)',
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

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # prepare db
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_elev['traj_interp'].apply(add_st_gh_contrib)

#%%
    if bool(distutils.util.strtobool(params.parametric)):
        spm_test = spm1d.stats.ttest
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest
        infer_params = {'force_iterations': True}

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    plot_utils.init_graphing(params.backend)
    plt.close('all')

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    fig = plt.figure(figsize=(110 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(3, 1)

    for i in range(3):
        style_axes(axs[i], 'Humerothoracic Elevation (Deg)' if i == 2 else None, 'Diff from HT Elevation (deg)')

    leg_mean = []
    max_pos = 140
    spm_y = [2, 2, 7]
    contrib_max_diff = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        traj_ht_euler_diff = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'euler.ht_isb', 1, 'up']), axis=0)
        traj_gh_elev_contrib = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        traj_st_elev_contrib = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        traj_gh_elev_euler_diff = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 1, 'up']), axis=0)
        traj_st_upward_euler_diff = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        traj_ht_contrib = traj_st_elev_contrib + traj_gh_elev_contrib
        traj_ht_euler_addition = traj_st_upward_euler_diff + traj_gh_elev_euler_diff

        ht_euler_ur_elev_diff = traj_ht_euler_diff - traj_ht_euler_addition
        ht_euler_contrib_diff = traj_ht_euler_diff - traj_ht_contrib

        # means and standard deviations
        ht_euler_add_mean = np.rad2deg(np.mean(ht_euler_ur_elev_diff, axis=0))
        ht_contrib_mean = np.rad2deg(np.mean(ht_euler_contrib_diff, axis=0))

        ht_euler_add_sd = np.rad2deg(np.std(ht_euler_ur_elev_diff, axis=0, ddof=1))
        ht_contrib_sd = np.rad2deg(np.std(ht_euler_contrib_diff, axis=0, ddof=1))

        # at max
        traj_ht_euler_diff_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['ht', 'euler.ht_isb', 1, 'up']), axis=0)
        traj_gh_elev_contrib_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'contribs', 1, 'up']), axis=0)
        traj_st_elev_contrib_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 1, 'up']), axis=0)
        traj_gh_elev_euler_diff_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_isb', 1, 'up']), axis=0)
        traj_st_upward_euler_diff_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 1, 'up']), axis=0)
        traj_ht_contrib_max = traj_st_elev_contrib_max + traj_gh_elev_contrib_max
        traj_ht_euler_addition_max = traj_st_upward_euler_diff_max + traj_gh_elev_euler_diff_max

        ht_euler_ur_elev_diff_max = traj_ht_euler_diff_max - traj_ht_euler_addition_max
        ht_euler_contrib_diff_max = traj_ht_euler_diff_max - traj_ht_contrib_max

        contrib_max_diff.append(max(np.rad2deg(np.max(np.abs(ht_euler_contrib_diff_max))),
                                    np.rad2deg(np.max(np.abs(ht_euler_contrib_diff)))))

        # means and standard deviations
        ht_euler_add_max_mean = np.rad2deg(np.mean(ht_euler_ur_elev_diff_max, axis=0))
        ht_contrib_max_mean = np.rad2deg(np.mean(ht_euler_contrib_diff_max, axis=0))

        ht_euler_add_max_sd = np.rad2deg(np.std(ht_euler_ur_elev_diff_max, axis=0, ddof=1))
        ht_contrib_max_sd = np.rad2deg(np.std(ht_euler_contrib_diff_max, axis=0, ddof=1))

        # spm
        ht_euler_diff_vs_euler_add = spm_test(ht_euler_ur_elev_diff, 0).inference(alpha, two_tailed=True,
                                                                                  **infer_params)
        # ht_euler_diff_vs_contribs = spm_test(ht_euler_contrib_diff, 0).inference(alpha, two_tailed=True,
        #                                                                          **infer_params)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        ht_euler_add_ln = mean_sd_plot(axs[cur_row], x, ht_euler_add_mean, ht_euler_add_sd,
                                       dict(color=color_map.colors[0], alpha=0.25),
                                       dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        ht_contribs_ln = mean_sd_plot(axs[cur_row], x, ht_contrib_mean, ht_contrib_sd,
                                      dict(color=color_map.colors[1], alpha=0.25),
                                      dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        # plot at max
        axs[cur_row].errorbar(max_pos, ht_euler_add_max_mean, yerr=ht_euler_add_max_sd, color=color_map.colors[0],
                              marker=markers[0], capsize=3)
        axs[cur_row].errorbar(max_pos - 3, ht_contrib_max_mean, yerr=ht_contrib_max_sd, color=color_map.colors[1],
                              marker=markers[1], capsize=3)

        # plot spm
        x_sig_euler_add = sig_filter(ht_euler_diff_vs_euler_add, x)
        # x_sig_contribs = sig_filter(ht_euler_diff_vs_contribs, x)
        axs[cur_row].plot(x_sig_euler_add, np.repeat(spm_y[cur_row], x_sig_euler_add.size), color=color_map.colors[0],
                          lw=2)
        # axs[cur_row].plot(x_sig_contribs, np.repeat(spm_y[cur_row] - 1, x_sig_contribs.size),
        #                   color=color_map.colors[1], lw=2)

        # print info
        print(activity)
        print('Euler Diff at Max: {:.2f}'.format(ht_euler_add_max_mean))
        print('Contrib Diff at Max: {:.2f}'.format(ht_contrib_max_mean))
        print('HT Elevation significance:')
        print(extract_sig(ht_euler_diff_vs_euler_add, x))
        print(output_spm_p(ht_euler_diff_vs_euler_add))

        if idx == 0:
            leg_mean.append(ht_euler_add_ln[0])
            leg_mean.append(ht_contribs_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('Difference from HT elevation', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.91)
    leg_left = fig.legend(leg_mean, ['ST UR + GH Elevation', 'ST + GH Contrib to Elevation'],
                          loc='lower left', bbox_to_anchor=(0., 0.93), ncol=2, handlelength=1.5, handletextpad=0.5,
                          columnspacing=0.75, labelspacing=0.3, borderpad=0.2)

    # set x ticks
    if int(params.min_elev) % 10 == 0:
        x_ticks_start = int(params.min_elev)
    else:
        x_ticks_start = int(params.min_elev) - int(params.min_elev) % 10

    if int(params.max_elev) % 10 == 0:
        x_ticks_end = int(params.max_elev)
    else:
        x_ticks_end = int(params.max_elev) + (10 - int(params.max_elev) % 10)
    x_ticks = np.arange(x_ticks_start, x_ticks_end + 1, 20)
    x_ticks = np.sort(np.concatenate((x_ticks, np.array([params.min_elev, params.max_elev]))))
    x_ticks = np.concatenate((x_ticks, [max_pos]))

    for i in range(3):
        axs[i].set_xticks(x_ticks)
        tick_labels = [str(i) for i in x_ticks]
        tick_labels[-1] = 'Max'
        axs[i].set_xticklabels(tick_labels)

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
