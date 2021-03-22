"""Compare differences by age group for contributions of the ST and GH joint to HT elevation using Euler angles
and contributions

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
dtheta_coarse: Incremental angle (deg) to use for coarse interpolation between minimum and maximum HT elevation analyzed.
min_elev: Minimum HT elevation angle (deg) utilized for analysis that encompasses all trials.
max_elev: Maximum HT elevation angle (deg) utilized for analysis that encompasses all trials.
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from functools import partial
    import numpy as np
    import matplotlib.pyplot as plt
    import spm1d
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.plot_utils import style_axes, mean_sd_plot, make_interactive, sig_filter
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare differences by age group for contributions of the ST and GH joint to HT '
                                     'elevation using Euler angles and contributions', __package__, __file__))
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
        spm_test = partial(spm1d.stats.ttest2, equal_var=False)
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest2
        infer_params = {'force_iterations': True}

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    plot_utils.init_graphing(params.backend)
    plt.close('all')

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    fig_st = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_st = fig_st.subplots(3, 2)

    fig_gh = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_gh = fig_gh.subplots(3, 2)

    st_lims = [(-5, 50), (-5, 50), (-5, 50)]
    gh_lims = [(-10, 95), (0, 90), (-10, 95)]
    for i in range(3):
        for j in range(2):
            style_axes(axs_st[i, j], 'Humerothoracic Elevation (Deg)' if i == 2 else None,
                       'Elevation (deg)' if j == 0 else None)
            style_axes(axs_gh[i, j], 'Humerothoracic Elevation (Deg)' if i == 2 else None,
                       'Elevation (deg)' if j == 0 else None)
            axs_st[i, j].set_ylim(st_lims[i][0], st_lims[i][1])
            axs_gh[i, j].set_ylim(gh_lims[i][0], gh_lims[i][1])
            axs_gh[i, j].yaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            axs_st[i, j].yaxis.set_major_locator(plticker.MultipleLocator(base=10.0))

    leg_mean_st = []
    leg_mean_gh = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        activity_df_lt35 = activity_df.loc[activity_df['age_group'] == '<35']
        activity_df_gt45 = activity_df.loc[activity_df['age_group'] == '>45']

        st_euler_lt35 = -np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'euler.st_isb', 1, 'up']), axis=0)
        st_contrib_lt35 = -np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        gh_euler_lt35 = -np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 1, 'up']), axis=0)
        gh_contrib_lt35 = -np.stack(activity_df_lt35['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 1, 'up']), axis=0)

        st_euler_gt45 = -np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'euler.st_isb', 1, 'up']), axis=0)
        st_contrib_gt45 = -np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        gh_euler_gt45 = -np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 1, 'up']), axis=0)
        gh_contrib_gt45 = -np.stack(activity_df_gt45['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 1, 'up']), axis=0)

        # means and standard deviations
        st_euler_mean_lt35 = np.rad2deg(np.mean(st_euler_lt35, axis=0))
        st_contrib_mean_lt35 = np.rad2deg(np.mean(st_contrib_lt35, axis=0))
        gh_euler_mean_lt35 = np.rad2deg(np.mean(gh_euler_lt35, axis=0))
        gh_contrib_mean_lt35 = np.rad2deg(np.mean(gh_contrib_lt35, axis=0))

        st_euler_mean_gt45 = np.rad2deg(np.mean(st_euler_gt45, axis=0))
        st_contrib_mean_gt45 = np.rad2deg(np.mean(st_contrib_gt45, axis=0))
        gh_euler_mean_gt45 = np.rad2deg(np.mean(gh_euler_gt45, axis=0))
        gh_contrib_mean_gt45 = np.rad2deg(np.mean(gh_contrib_gt45, axis=0))

        st_euler_sd_lt35 = np.rad2deg(np.std(st_euler_lt35, ddof=1, axis=0))
        st_contrib_sd_lt35 = np.rad2deg(np.std(st_contrib_lt35, ddof=1, axis=0))
        gh_euler_sd_lt35 = np.rad2deg(np.std(gh_euler_lt35, ddof=1, axis=0))
        gh_contrib_sd_lt35 = np.rad2deg(np.std(gh_contrib_lt35, ddof=1, axis=0))

        st_euler_sd_gt45 = np.rad2deg(np.std(st_euler_gt45, ddof=1, axis=0))
        st_contrib_sd_gt45 = np.rad2deg(np.std(st_contrib_gt45, ddof=1, axis=0))
        gh_euler_sd_gt45 = np.rad2deg(np.std(gh_euler_gt45, ddof=1, axis=0))
        gh_contrib_sd_gt45 = np.rad2deg(np.std(gh_contrib_gt45, ddof=1, axis=0))

        # spm
        st_euler_spm = spm_test(st_euler_lt35, st_euler_gt45, equal_var=False).inference(alpha, two_tailed=True,
                                                                                         **infer_params)
        st_contrib_spm = spm_test(st_contrib_lt35, st_contrib_gt45, equal_var=False).inference(alpha, two_tailed=True,
                                                                                               **infer_params)
        gh_euler_spm = spm_test(gh_euler_lt35, gh_euler_gt45, equal_var=False).inference(alpha, two_tailed=True,
                                                                                         **infer_params)
        gh_contrib_spm = spm_test(gh_contrib_lt35, gh_contrib_gt45, equal_var=False).inference(alpha, two_tailed=True,
                                                                                               **infer_params)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        st_euler_ln_lt35 = mean_sd_plot(axs_st[cur_row, 0], x, st_euler_mean_lt35, st_euler_sd_lt35,
                                        dict(color=color_map.colors[0], alpha=0.25),
                                        dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_contrib_ln_lt35 = mean_sd_plot(axs_st[cur_row, 1], x, st_contrib_mean_lt35, st_contrib_sd_lt35,
                                          dict(color=color_map.colors[2], alpha=0.25),
                                          dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        gh_euler_ln_lt35 = mean_sd_plot(axs_gh[cur_row, 0], x, gh_euler_mean_lt35, gh_euler_sd_lt35,
                                        dict(color=color_map.colors[0], alpha=0.25),
                                        dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        gh_contrib_ln_lt35 = mean_sd_plot(axs_gh[cur_row, 1], x, gh_contrib_mean_lt35, gh_contrib_sd_lt35,
                                          dict(color=color_map.colors[2], alpha=0.25),
                                          dict(color=color_map.colors[2], marker=markers[2], markevery=20))

        st_euler_ln_gt45 = mean_sd_plot(axs_st[cur_row, 0], x, st_euler_mean_gt45, st_euler_sd_gt45,
                                        dict(color=color_map.colors[1], alpha=0.25),
                                        dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        st_contrib_ln_gt45 = mean_sd_plot(axs_st[cur_row, 1], x, st_contrib_mean_gt45, st_contrib_sd_gt45,
                                          dict(color=color_map.colors[3], alpha=0.25),
                                          dict(color=color_map.colors[3], marker=markers[3], markevery=20))
        gh_euler_ln_gt45 = mean_sd_plot(axs_gh[cur_row, 0], x, gh_euler_mean_gt45, gh_euler_sd_gt45,
                                        dict(color=color_map.colors[1], alpha=0.25),
                                        dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        gh_contrib_ln_gt45 = mean_sd_plot(axs_gh[cur_row, 1], x, gh_contrib_mean_gt45, gh_contrib_sd_gt45,
                                          dict(color=color_map.colors[3], alpha=0.25),
                                          dict(color=color_map.colors[3], marker=markers[3], markevery=20))

        # plot spm
        st_euler_x_sig = sig_filter(st_euler_spm, x)
        st_contrib_x_sig = sig_filter(st_contrib_spm, x)
        gh_euler_x_sig = sig_filter(gh_euler_spm, x)
        gh_contrib_x_sig = sig_filter(gh_contrib_spm, x)
        axs_st[cur_row, 0].plot(st_euler_x_sig, np.repeat(st_lims[cur_row][1], st_euler_x_sig.size), color='k', lw=2)
        axs_st[cur_row, 1].plot(st_contrib_x_sig, np.repeat(st_lims[cur_row][1], st_euler_x_sig.size), color='k', lw=2)
        axs_gh[cur_row, 0].plot(gh_euler_x_sig, np.repeat(gh_lims[cur_row][1], gh_euler_x_sig.size), color='k', lw=2)
        axs_gh[cur_row, 1].plot(gh_contrib_x_sig, np.repeat(gh_lims[cur_row][1], gh_euler_x_sig.size), color='k', lw=2)

        if idx == 0:
            leg_mean_st.append(st_euler_ln_lt35[0])
            leg_mean_st.append(st_euler_ln_gt45[0])
            leg_mean_st.append(st_contrib_ln_lt35[0])
            leg_mean_st.append(st_contrib_ln_gt45[0])
            leg_mean_gh.append(gh_euler_ln_lt35[0])
            leg_mean_gh.append(gh_euler_ln_gt45[0])
            leg_mean_gh.append(gh_contrib_ln_lt35[0])
            leg_mean_gh.append(gh_contrib_ln_gt45[0])

    for idx, (fig, leg_entries) in enumerate(zip((fig_st, fig_gh), (leg_mean_st, leg_mean_gh))):
        # figure title and legend
        plt.figure(fig.number)
        plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
        fig.suptitle(('ST' if idx == 0 else 'GH') + ' Elevation Comparison', x=0.47, y=0.99, fontweight='bold')
        plt.subplots_adjust(top=0.93)
        leg_left = fig.legend(leg_entries[:2], ['Euler <35', 'Euler >45'],
                              loc='upper left', bbox_to_anchor=(0, 1), ncol=1, handlelength=1.5, handletextpad=0.5,
                              columnspacing=0.75, labelspacing=0.3, borderpad=0.2)
        leg_right = fig.legend(leg_entries[2:], ['Contrib <35', 'Contrib >45'],
                               loc='upper right', bbox_to_anchor=(1, 1), ncol=1, handlelength=1.5, handletextpad=0.5,
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
        for row in axs_st:
            for ax in row:
                ax.set_xticks(x_ticks)

        # add axes titles
        _, y0, _, h = axs_st[0, 0].get_position().bounds
        fig_st.text(0.5, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_st[1, 0].get_position().bounds
        fig_st.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

        _, y0, _, h = axs_st[2, 0].get_position().bounds
        fig_st.text(0.5, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

        make_interactive()

    plt.show()
