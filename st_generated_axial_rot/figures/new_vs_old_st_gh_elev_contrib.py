"""Compare contributions of the ST and GH joint to HT elevation using Euler angles and contributions

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

    config_dir = Path(mod_arg_parser('Compare contributions of the ST and GH joint to HT elevation using Euler angles '
                                     'and contributions', __package__, __file__))
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

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(3, 2)

    for i in range(3):
        for j in range(2):
            style_axes(axs[i, j], 'Humerothoracic Elevation (Deg)' if i == 2 else None,
                       'Elevation (deg)' if j == 0 else None)
            axs[i, j].set_ylim(-60 if j == 0 else -110, 5)

    leg_mean = []
    max_pos = 140
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        st_euler = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'euler.st_isb', 1, 'up']), axis=0)
        st_contrib = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        gh_euler = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 1, 'up']), axis=0)
        gh_contrib = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 1, 'up']), axis=0)

        st_euler_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'euler.st_isb', 1, 'up']), axis=0)
        st_contrib_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 1, 'up']), axis=0)
        gh_euler_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_isb', 1, 'up']), axis=0)
        gh_contrib_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'contribs', 1, 'up']), axis=0)

        # means and standard deviations
        st_euler_mean = np.rad2deg(np.mean(st_euler, axis=0))
        st_contrib_mean = np.rad2deg(np.mean(st_contrib, axis=0))
        gh_euler_mean = np.rad2deg(np.mean(gh_euler, axis=0))
        gh_contrib_mean = np.rad2deg(np.mean(gh_contrib, axis=0))

        st_euler_max_mean = np.rad2deg(np.mean(st_euler_max, axis=0))
        st_contrib_max_mean = np.rad2deg(np.mean(st_contrib_max, axis=0))
        gh_euler_max_mean = np.rad2deg(np.mean(gh_euler_max, axis=0))
        gh_contrib_max_mean = np.rad2deg(np.mean(gh_contrib_max, axis=0))

        st_euler_sd = np.rad2deg(np.std(st_euler, ddof=1, axis=0))
        st_contrib_sd = np.rad2deg(np.std(st_contrib, ddof=1, axis=0))
        gh_euler_sd = np.rad2deg(np.std(gh_euler, ddof=1, axis=0))
        gh_contrib_sd = np.rad2deg(np.std(gh_contrib, ddof=1, axis=0))

        st_euler_max_sd = np.rad2deg(np.std(st_euler_max, ddof=1, axis=0))
        st_contrib_max_sd = np.rad2deg(np.std(st_contrib_max, ddof=1, axis=0))
        gh_euler_max_sd = np.rad2deg(np.std(gh_euler_max, ddof=1, axis=0))
        gh_contrib_max_sd = np.rad2deg(np.std(gh_contrib_max, ddof=1, axis=0))

        # spm
        st_spm = spm_test(st_euler, st_contrib).inference(alpha, two_tailed=True, **infer_params)
        gh_spm = spm_test(gh_euler, gh_contrib).inference(alpha, two_tailed=True, **infer_params)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        st_euler_ln = mean_sd_plot(axs[cur_row, 0], x, st_euler_mean, st_euler_sd,
                                   dict(color=color_map.colors[0], alpha=0.25),
                                   dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_contrib_ln = mean_sd_plot(axs[cur_row, 0], x, st_contrib_mean, st_contrib_sd,
                                     dict(color=color_map.colors[1], alpha=0.25),
                                     dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        gh_euler_ln = mean_sd_plot(axs[cur_row, 1], x, gh_euler_mean, gh_euler_sd,
                                   dict(color=color_map.colors[0], alpha=0.25),
                                   dict(color=color_map.colors[0], marker=markers[2], markevery=20))
        gh_contrib_ln = mean_sd_plot(axs[cur_row, 1], x, gh_contrib_mean, gh_contrib_sd,
                                     dict(color=color_map.colors[1], alpha=0.25),
                                     dict(color=color_map.colors[1], marker=markers[3], markevery=20))

        axs[cur_row, 0].errorbar(max_pos - 3, st_euler_max_mean, yerr=st_euler_max_sd, color=color_map.colors[0],
                                 marker=markers[0], capsize=3)
        axs[cur_row, 0].errorbar(max_pos, st_contrib_max_mean, yerr=st_contrib_max_sd, color=color_map.colors[1],
                                 marker=markers[1], capsize=3)

        axs[cur_row, 1].errorbar(max_pos - 3, gh_euler_max_mean, yerr=gh_euler_max_sd, color=color_map.colors[0],
                                 marker=markers[0], capsize=3)
        axs[cur_row, 1].errorbar(max_pos, gh_contrib_max_mean, yerr=gh_contrib_max_sd, color=color_map.colors[1],
                                 marker=markers[1], capsize=3)

        # plot spm
        st_x_sig = sig_filter(st_spm, x)
        gh_x_sig = sig_filter(gh_spm, x)
        axs[cur_row, 0].plot(st_x_sig, np.repeat(2, st_x_sig.size), color='k', lw=2)
        axs[cur_row, 1].plot(gh_x_sig, np.repeat(0, gh_x_sig.size), color='k', lw=2)
        print(activity)
        print('Mean Diff at Max ST: {:.2f}'.format(st_contrib_max_mean - st_euler_max_mean))
        print('Mean Diff at Max GH: {:.2f}'.format(gh_contrib_max_mean - gh_euler_max_mean))
        print('ST HT Elevation angles')
        print(extract_sig(st_spm, x))
        print(output_spm_p(st_spm))
        print('GH HT Elevation angles')
        print(extract_sig(gh_spm, x))
        print(output_spm_p(gh_spm))

        if idx == 0:
            leg_mean.append(st_euler_ln[0])
            leg_mean.append(st_contrib_ln[0])
            leg_mean.append(gh_euler_ln[0])
            leg_mean.append(gh_contrib_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('Comparison between Euler Angles and\nST/GH Contribution for Measuring Elevation',
                 x=0.47, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    leg_left = fig.legend(leg_mean[:2], ['ST Upward Rotation', 'ST Contribution\nto Elevation'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          labelspacing=0.3, borderpad=0.2)
    leg_right = fig.legend(leg_mean[2:], ['GH Elevation', 'GH Contribution\nto Elevation'], loc='upper right',
                           bbox_to_anchor=(1, 1), ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                           labelspacing=0.3, borderpad=0.2)

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
    for row in axs:
        for ax in row:
            ax.set_xticks(x_ticks)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
