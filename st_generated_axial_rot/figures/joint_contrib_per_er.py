"""Compare contributions of the ST and GH joints towards Elevation, Axial Rotation, and PoE for external rotation trials

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
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
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare contributions of the ST and GH joints towards Elevation, Axial Rotation, '
                                     'and PoE for external rotation trials', __package__, __file__))
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
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'erar': 0, 'era90': 1}

    x = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(2, 2)

    # style axes, add x and y labels
    for i in range(2):
        for j in range(2):
            style_axes(axs[i, j], 'Motion Completion (%)' if i == 1 else None,
                       'Motion Allocation (%)' if j == 0 else None)
            axs[i, j].yaxis.set_major_locator(plticker.MultipleLocator(10))
            axs[i, j].set_ylim(0, 100)

    # plot
    max_pos = 140
    leg_mean = []
    # leg_patch_t = []
    # alpha_patch = []
    for idx, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        st_contribs = np.stack(activity_df['st_contribs_interp'], axis=0)
        gh_contribs = np.stack(activity_df['gh_contribs_interp'], axis=0)
        trajs_st_elev = st_contribs[:, 1:, 1]
        trajs_gh_elev = gh_contribs[:, 1:, 1]

        trajs_st_axial = st_contribs[:, 1:, 2]
        trajs_gh_axial = gh_contribs[:, 1:, 2]

        trajs_st_poe = st_contribs[:, 1:, 0]
        trajs_gh_poe = gh_contribs[:, 1:, 0]

        trajs_st_total = np.abs(trajs_st_poe) + np.abs(trajs_st_elev) + np.abs(trajs_st_axial)
        trajs_gh_total = np.abs(trajs_gh_poe) + np.abs(trajs_gh_elev) + np.abs(trajs_gh_axial)

        st_elev_per = (np.abs(trajs_st_elev) / trajs_st_total) * 100
        st_axial_per = (np.abs(trajs_st_axial) / trajs_st_total) * 100
        st_poe_per = (np.abs(trajs_st_poe) / trajs_st_total) * 100

        gh_elev_per = (np.abs(trajs_gh_elev) / trajs_gh_total) * 100
        gh_axial_per = (np.abs(trajs_gh_axial) / trajs_gh_total) * 100
        gh_poe_per = (np.abs(trajs_gh_poe) / trajs_gh_total) * 100

        # means and standard deviations
        st_elev_mean = np.mean(st_elev_per, axis=0)
        st_axial_mean = np.mean(st_axial_per, axis=0)
        st_poe_mean = np.mean(st_poe_per, axis=0)
        gh_elev_mean = np.mean(gh_elev_per, axis=0)
        gh_axial_mean = np.mean(gh_axial_per, axis=0)
        gh_poe_mean = np.mean(gh_poe_per, axis=0)

        st_elev_sd = np.std(st_elev_per, axis=0, ddof=1)
        st_axial_sd = np.std(st_axial_per, axis=0, ddof=1)
        st_poe_sd = np.std(st_poe_per, axis=0, ddof=1)
        gh_elev_sd = np.std(gh_elev_per, axis=0, ddof=1)
        gh_axial_sd = np.std(gh_axial_per, axis=0, ddof=1)
        gh_poe_sd = np.std(gh_poe_per, axis=0, ddof=1)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        st_elev_ln = mean_sd_plot(axs[cur_row, 0], x[1:], st_elev_mean, st_elev_sd,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_axial_ln = mean_sd_plot(axs[cur_row, 0], x[1:], st_axial_mean, st_axial_sd,
                                   dict(color=color_map.colors[1], alpha=0.25),
                                   dict(color=color_map.colors[1], marker=markers[0], markevery=20))
        st_poe_ln = mean_sd_plot(axs[cur_row, 0], x[1:], st_poe_mean, st_poe_sd,
                                 dict(color=color_map.colors[2], alpha=0.25),
                                 dict(color=color_map.colors[2], marker=markers[0], markevery=20))

        gh_elev_ln = mean_sd_plot(axs[cur_row, 1], x[1:], gh_elev_mean, gh_elev_sd,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[1], markevery=20))
        gh_axial_ln = mean_sd_plot(axs[cur_row, 1], x[1:], gh_axial_mean, gh_axial_sd,
                                   dict(color=color_map.colors[1], alpha=0.25),
                                   dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        gh_poe_ln = mean_sd_plot(axs[cur_row, 1], x[1:], gh_poe_mean, gh_poe_sd,
                                 dict(color=color_map.colors[2], alpha=0.25),
                                 dict(color=color_map.colors[2], marker=markers[1], markevery=20))

        # plot title
        axs[cur_row, 0].set_title('ST Joint', y=0.95)
        axs[cur_row, 1].set_title('GH Joint', y=0.95)

        # print percentages at maximum
        print(activity)
        print('ST Elevation {:.2f}'.format(st_elev_mean[-1]))
        print('ST Axial Rotation {:.2f}'.format(st_axial_mean[-1]))
        print('ST PoE {:.2f}'.format(st_poe_mean[-1]))

        print('GH Elevation {:.2f}'.format(gh_elev_mean[-1]))
        print('GH Axial Rotation {:.2f}'.format(gh_axial_mean[-1]))
        print('GH PoE {:.2f}'.format(gh_poe_mean[-1]))

        if idx == 0:
            leg_mean.append(st_elev_ln[0])
            leg_mean.append(st_axial_ln[0])
            leg_mean.append(st_poe_ln[0])
            leg_mean.append(gh_elev_ln[0])
            leg_mean.append(gh_axial_ln[0])
            leg_mean.append(gh_poe_ln[0])

    # figure title and legend
    plt.figure(fig.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('ST and GH Joint Motion Allocation Percentages', x=0.47, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend(leg_mean[:3], ['Elevation', 'Axial Rotation', 'PoE'], loc='upper left', bbox_to_anchor=(0, 1),
                          ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, labelspacing=0.3,
                          borderpad=0.2)
    leg_right = fig.legend(leg_mean[3:], ['Elevation', 'Axial Rotation', 'PoE'], loc='upper right',
                           bbox_to_anchor=(1, 1), ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                           labelspacing=0.3, borderpad=0.2)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'External Rotation in Adduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'External Rotation in 90° of Abduction', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
