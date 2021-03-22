"""Contributions of ST and GH joints to PoE, elevation, and axial rotation for external rotation trials by axis

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
era90_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in 90 deg of abduction trials.
erar_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in adduction trials.
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Contributions of ST and GH joints to PoE, elevation, and axial rotation for '
                                     'external rotation trials by axis',
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

    db_er_endpts = ready_er_db(db, params.torso_def, params.scap_lateral, params.erar_endpts, params.era90_endpts,
                               params.dtheta_fine)

    #%%
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'erar': 0, 'era90': 1}

    x = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(3, 2)

    # style axes, add x and y labels
    for i in range(3):
        for j in range(2):
            style_axes(axs[i, j], 'Motion Completion (%)' if i == 2 else None,
                       'Rotation (deg)' if j == 0 else None)
            axs[i, j].yaxis.set_major_locator(plticker.MultipleLocator(10))

    # plot
    leg_mean = []
    max_pos = 140
    for idx, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        st_contribs = np.stack(activity_df['st_contribs_interp'], axis=0)
        gh_contribs = np.stack(activity_df['gh_contribs_interp'], axis=0)
        trajs_st_elev = st_contribs[:, :, 1]
        trajs_gh_elev = gh_contribs[:, :, 1]

        trajs_st_axial = st_contribs[:, :, 2]
        trajs_gh_axial = gh_contribs[:, :, 2]

        trajs_st_poe = st_contribs[:, :, 0]
        trajs_gh_poe = gh_contribs[:, :, 0]

        # means and standard deviations
        st_elev_mean = np.rad2deg(np.mean(trajs_st_elev, axis=0))
        gh_elev_mean = np.rad2deg(np.mean(trajs_gh_elev, axis=0))
        st_axial_mean = np.rad2deg(np.mean(trajs_st_axial, axis=0))
        gh_axial_mean = np.rad2deg(np.mean(trajs_gh_axial, axis=0))
        st_poe_mean = np.rad2deg(np.mean(trajs_st_poe, axis=0))
        gh_poe_mean = np.rad2deg(np.mean(trajs_gh_poe, axis=0))

        st_elev_sd = np.rad2deg(np.std(trajs_st_elev, axis=0, ddof=1))
        gh_elev_sd = np.rad2deg(np.std(trajs_gh_elev, axis=0, ddof=1))
        st_axial_sd = np.rad2deg(np.std(trajs_st_axial, axis=0, ddof=1))
        gh_axial_sd = np.rad2deg(np.std(trajs_gh_axial, axis=0, ddof=1))
        st_poe_sd = np.rad2deg(np.std(trajs_st_poe, axis=0, ddof=1))
        gh_poe_sd = np.rad2deg(np.std(trajs_gh_poe, axis=0, ddof=1))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        st_elev_ln = mean_sd_plot(axs[0, cur_row], x, st_elev_mean, st_elev_sd,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        gh_elev_ln = mean_sd_plot(axs[0, cur_row], x, gh_elev_mean, gh_elev_sd,
                                  dict(color=color_map.colors[1], alpha=0.25),
                                  dict(color=color_map.colors[1], marker=markers[0], markevery=20))
        st_axial_ln = mean_sd_plot(axs[1, cur_row], x, st_axial_mean, st_axial_sd,
                                   dict(color=color_map.colors[0], alpha=0.25),
                                   dict(color=color_map.colors[0], marker=markers[1], markevery=20))
        gh_axial_ln = mean_sd_plot(axs[1, cur_row], x, gh_axial_mean, gh_axial_sd,
                                   dict(color=color_map.colors[1], alpha=0.25),
                                   dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        st_poe_ln = mean_sd_plot(axs[2, cur_row], x, st_poe_mean, st_poe_sd,
                                 dict(color=color_map.colors[0], alpha=0.25),
                                 dict(color=color_map.colors[0], marker=markers[2], markevery=20))
        gh_poe_ln = mean_sd_plot(axs[2, cur_row], x, gh_poe_mean, gh_poe_sd,
                                 dict(color=color_map.colors[1], alpha=0.25),
                                 dict(color=color_map.colors[1], marker=markers[2], markevery=20))

        if idx == 0:
            leg_mean.append(st_elev_ln[0])
            leg_mean.append(st_axial_ln[0])
            leg_mean.append(st_poe_ln[0])
            leg_mean.append(gh_elev_ln[0])
            leg_mean.append(gh_axial_ln[0])
            leg_mean.append(gh_poe_ln[0])

    axs[0, 0].set_title('ER-ADD', y=0.8)
    axs[0, 1].set_title('ER-ABD', y=0.8)

    # figure title and legend
    plt.figure(fig.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=-1.5)
    fig.suptitle('ST and GH Joint Rotational Contributions', x=0.47, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend(leg_mean[:3], ['ST Elevation', 'ST Axial Rotation', 'ST PoE'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          labelspacing=0.3, borderpad=0.2)
    leg_right = fig.legend(leg_mean[3:], ['GH Elevation', 'GH Axial Rotation', 'GH PoE'], loc='upper right',
                           bbox_to_anchor=(1, 1), ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                           labelspacing=0.3, borderpad=0.2)

    # add arrows indicating direction
    # axs[0, 0].arrow(35, -15, 0, -15, length_includes_head=True, head_width=2, head_length=2)
    # axs[0, 0].text(23, -15, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Elevation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Plane of Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    plt.show()
