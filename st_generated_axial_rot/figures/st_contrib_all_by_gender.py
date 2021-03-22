"""ST contributions to HT axial rotation by gender

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
era90_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in 90 deg of abduction trials.
erar_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in adduction trials.
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
    from functools import partial
    import numpy as np
    import matplotlib.pyplot as plt
    import spm1d
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes,
                                                          sig_filter, extract_sig)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('ST contributions to HT axial rotation by gender',
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

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_elev['traj_interp'].apply(add_st_gh_contrib)
    db_er_endpts = ready_er_db(db, params.torso_def, params.scap_lateral, params.erar_endpts, params.era90_endpts,
                               params.dtheta_fine)

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
    max_pos = 140

    x_elev = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    x_er = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    gs = fig.add_gridspec(6, 2)
    axs_elev = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[2:4, 0]), fig.add_subplot(gs[4:6, 0])]
    axs_er = [fig.add_subplot(gs[:3, 1]), fig.add_subplot(gs[3:6, 1])]

    for i in range(3):
        style_axes(axs_elev[i], 'Humerothoracic Elevation (Deg)' if i == 2 else None, 'ST Axial Rotation (Deg)')

    for i in range(2):
        style_axes(axs_er[i], 'Motion Completion (%)' if i == 1 else None, 'ST Axial Rotation (Deg)')

    # plot elevation
    leg_elev_mean = []
    spm_y = {'ca': 2, 'sa': 4, 'fe': 21}
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        m_df = activity_df[activity_df['Gender'] == 'M']
        f_df = activity_df[activity_df['Gender'] == 'F']

        traj_st_m = np.stack(m_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 2, 'up']), axis=0)
        traj_st_f = np.stack(f_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 2, 'up']), axis=0)

        traj_st_m_max = np.stack(m_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0)
        traj_st_f_max = np.stack(f_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0)

        # means and standard deviations
        st_m_mean = np.rad2deg(np.mean(traj_st_m, axis=0))
        st_f_mean = np.rad2deg(np.mean(traj_st_f, axis=0))
        st_m_max_mean = np.rad2deg(np.mean(traj_st_m_max, axis=0))
        st_f_max_mean = np.rad2deg(np.mean(traj_st_f_max, axis=0))

        st_m_sd = np.rad2deg(np.std(traj_st_m, axis=0, ddof=1))
        st_f_sd = np.rad2deg(np.std(traj_st_f, axis=0, ddof=1))
        st_m_max_sd = np.rad2deg(np.std(traj_st_m_max, axis=0, ddof=1))
        st_f_max_sd = np.rad2deg(np.std(traj_st_f_max, axis=0, ddof=1))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        st_m_ln = mean_sd_plot(axs_elev[cur_row], x_elev, st_m_mean, st_m_sd,
                               dict(color=color_map.colors[0], alpha=0.25),
                               dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_f_ln = mean_sd_plot(axs_elev[cur_row], x_elev, st_f_mean, st_f_sd,
                               dict(color=color_map.colors[1], alpha=0.25),
                               dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        axs_elev[cur_row].errorbar(max_pos, st_m_max_mean, yerr=st_m_max_sd, color=color_map.colors[0],
                                   marker=markers[0], capsize=3)
        axs_elev[cur_row].errorbar(max_pos, st_f_max_mean, yerr=st_f_max_sd, color=color_map.colors[1],
                                   marker=markers[1], capsize=3)

        # spm
        m_vs_f = spm_test(traj_st_m, traj_st_f).inference(alpha, two_tailed=True, **infer_params)
        x_sig = sig_filter(m_vs_f, x_elev)
        axs_elev[cur_row].plot(x_sig, np.repeat(spm_y[activity.lower()], x_sig.size), color='k')
        print(activity)
        print(extract_sig(m_vs_f, x_elev))

        if idx == 0:
            leg_elev_mean.append(st_m_ln[0])
            leg_elev_mean.append(st_f_ln[0])

    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        m_df = activity_df[activity_df['Gender'] == 'M']
        f_df = activity_df[activity_df['Gender'] == 'F']

        traj_st_m = np.stack(m_df['st_contribs_interp'], axis=0)[:, :, 2]
        traj_st_f = np.stack(f_df['st_contribs_interp'], axis=0)[:, :, 2]

        # means and standard deviations
        st_m_mean = np.rad2deg(np.mean(traj_st_m, axis=0))
        st_f_mean = np.rad2deg(np.mean(traj_st_f, axis=0))
        st_m_sd = np.rad2deg(np.std(traj_st_m, axis=0, ddof=1))
        st_f_sd = np.rad2deg(np.std(traj_st_f, axis=0, ddof=1))

        st_m_ln = mean_sd_plot(axs_er[idx_act], x_er, st_m_mean, st_m_sd,
                               dict(color=color_map.colors[0], alpha=0.25),
                               dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_f_ln = mean_sd_plot(axs_er[idx_act], x_er, st_f_mean, st_f_sd,
                               dict(color=color_map.colors[1], alpha=0.25),
                               dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        m_vs_f = spm_test(traj_st_m[:, 1:], traj_st_f[:, 1:]).inference(alpha, two_tailed=True,
                                                                        **infer_params)
        x_sig = sig_filter(m_vs_f, x_er[1:])
        axs_er[idx_act].plot(x_sig, np.repeat(2, x_sig.size), color='k')
        print(activity)
        print(extract_sig(m_vs_f, x_er[1:]))

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('ST Contributions to HT Axial Rotation by Gender', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend(leg_elev_mean, ['M', 'F'], loc='upper left', bbox_to_anchor=(0, 1),
                          ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, labelspacing=0.3,
                          borderpad=0.2)

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
        axs_elev[i].set_xticks(x_ticks)
        tick_labels = [str(i) for i in x_ticks]
        tick_labels[-1] = 'Max'
        axs_elev[i].set_xticklabels(tick_labels)

    # add arrows indicating direction
    axs_elev[0].arrow(35, -15, 0, -10, length_includes_head=True, head_width=2, head_length=2)
    axs_elev[0].text(23, -15, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs_elev[0].get_position().bounds
    fig.text(0.25, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_elev[1].get_position().bounds
    fig.text(0.25, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_elev[2].get_position().bounds
    fig.text(0.25, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_er[0].get_position().bounds
    fig.text(0.75, y0 + h * 1.01, 'External Rotation in Adduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_er[1].get_position().bounds
    fig.text(0.77, y0 + h * 1.01, r'External Rotation in 90° of Abduction', ha='center', fontsize=11,
             fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
