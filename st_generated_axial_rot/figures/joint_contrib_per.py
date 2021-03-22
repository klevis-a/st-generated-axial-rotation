"""Compare contributions of the ST and GH joints towards Elevation, Axial Rotation, and PoE for elevation trials

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
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare contributions of the ST and GH joints towards Elevation, Axial Rotation, '
                                     'and PoE for elevation trials', __package__, __file__))
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

    #%%
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(3, 2)

    # style axes, add x and y labels
    for i in range(3):
        for j in range(2):
            style_axes(axs[i, j], 'Humerothoracic Elevation (Deg)' if i == 2 else None,
                       'Motion Allocation (%)' if j == 0 else None)
            axs[i, j].yaxis.set_major_locator(plticker.MultipleLocator(10))
            axs[i, j].set_ylim(0, 100)

    # plot
    max_pos = 140
    leg_mean = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        trajs_st_elev = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        trajs_gh_elev = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 1, 'up']), axis=0)

        trajs_st_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 2, 'up']), axis=0)
        trajs_gh_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 2, 'up']), axis=0)

        trajs_st_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 0, 'up']), axis=0)
        trajs_gh_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 0, 'up']), axis=0)

        # subject o45_009 has a brief discontinuity at the start so I want to exclude those data points
        is_o45_009 = (activity_df['Subject_Name'] == 'O45_009_F_63_R').to_numpy()
        trajs_st_elev[is_o45_009, :10] = np.nan
        trajs_gh_elev[is_o45_009, :10] = np.nan
        trajs_st_axial[is_o45_009, :10] = np.nan
        trajs_gh_axial[is_o45_009, :10] = np.nan
        trajs_st_poe[is_o45_009, :10] = np.nan
        trajs_gh_poe[is_o45_009, :10] = np.nan

        trajs_st_total = np.abs(trajs_st_poe) + np.abs(trajs_st_elev) + np.abs(trajs_st_axial)
        trajs_gh_total = np.abs(trajs_gh_poe) + np.abs(trajs_gh_elev) + np.abs(trajs_gh_axial)

        st_elev_per = (np.abs(trajs_st_elev) / trajs_st_total) * 100
        st_axial_per = (np.abs(trajs_st_axial) / trajs_st_total) * 100
        st_poe_per = (np.abs(trajs_st_poe) / trajs_st_total) * 100

        gh_elev_per = (np.abs(trajs_gh_elev) / trajs_gh_total) * 100
        gh_axial_per = (np.abs(trajs_gh_axial) / trajs_gh_total) * 100
        gh_poe_per = (np.abs(trajs_gh_poe) / trajs_gh_total) * 100

        # means and standard deviations
        st_elev_mean = np.nanmean(st_elev_per, axis=0)
        st_axial_mean = np.nanmean(st_axial_per, axis=0)
        st_poe_mean = np.nanmean(st_poe_per, axis=0)
        gh_elev_mean = np.nanmean(gh_elev_per, axis=0)
        gh_axial_mean = np.nanmean(gh_axial_per, axis=0)
        gh_poe_mean = np.nanmean(gh_poe_per, axis=0)

        st_elev_sd = np.nanstd(st_elev_per, axis=0, ddof=1)
        st_axial_sd = np.nanstd(st_axial_per, axis=0, ddof=1)
        st_poe_sd = np.nanstd(st_poe_per, axis=0, ddof=1)
        gh_elev_sd = np.nanstd(gh_elev_per, axis=0, ddof=1)
        gh_axial_sd = np.nanstd(gh_axial_per, axis=0, ddof=1)
        gh_poe_sd = np.nanstd(gh_poe_per, axis=0, ddof=1)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        st_elev_ln = mean_sd_plot(axs[cur_row, 0], x, st_elev_mean, st_elev_sd,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_axial_ln = mean_sd_plot(axs[cur_row, 0], x, st_axial_mean, st_axial_sd,
                                   dict(color=color_map.colors[1], alpha=0.25),
                                   dict(color=color_map.colors[1], marker=markers[0], markevery=20))
        st_poe_ln = mean_sd_plot(axs[cur_row, 0], x, st_poe_mean, st_poe_sd,
                                 dict(color=color_map.colors[2], alpha=0.25),
                                 dict(color=color_map.colors[2], marker=markers[0], markevery=20))

        gh_elev_ln = mean_sd_plot(axs[cur_row, 1], x, gh_elev_mean, gh_elev_sd,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[1], markevery=20))
        gh_axial_ln = mean_sd_plot(axs[cur_row, 1], x, gh_axial_mean, gh_axial_sd,
                                   dict(color=color_map.colors[1], alpha=0.25),
                                   dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        gh_poe_ln = mean_sd_plot(axs[cur_row, 1], x, gh_poe_mean, gh_poe_sd,
                                 dict(color=color_map.colors[2], alpha=0.25),
                                 dict(color=color_map.colors[2], marker=markers[1], markevery=20))

        # at maximum
        trajs_st_elev_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 1, 'up']), axis=0)
        trajs_gh_elev_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'contribs', 1, 'up']), axis=0)

        trajs_st_axial_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0)
        trajs_gh_axial_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'contribs', 2, 'up']), axis=0)

        trajs_st_poe_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 0, 'up']), axis=0)
        trajs_gh_poe_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'contribs', 0, 'up']), axis=0)

        trajs_st_total_max = np.abs(trajs_st_poe_max) + np.abs(trajs_st_elev_max) + np.abs(trajs_st_axial_max)
        trajs_gh_total_max = np.abs(trajs_gh_poe_max) + np.abs(trajs_gh_elev_max) + np.abs(trajs_gh_axial_max)

        st_elev_per_max = (np.abs(trajs_st_elev_max) / trajs_st_total_max) * 100
        st_axial_per_max = (np.abs(trajs_st_axial_max) / trajs_st_total_max) * 100
        st_poe_per_max = (np.abs(trajs_st_poe_max) / trajs_st_total_max) * 100

        gh_elev_per_max = (np.abs(trajs_gh_elev_max) / trajs_gh_total_max) * 100
        gh_axial_per_max = (np.abs(trajs_gh_axial_max) / trajs_gh_total_max) * 100
        gh_poe_per_max = (np.abs(trajs_gh_poe_max) / trajs_gh_total_max) * 100

        # means and standard deviations
        st_elev_mean_max = np.mean(st_elev_per_max, axis=0)
        st_axial_mean_max = np.mean(st_axial_per_max, axis=0)
        st_poe_mean_max = np.mean(st_poe_per_max, axis=0)
        gh_elev_mean_max = np.mean(gh_elev_per_max, axis=0)
        gh_axial_mean_max = np.mean(gh_axial_per_max, axis=0)
        gh_poe_mean_max = np.mean(gh_poe_per_max, axis=0)

        st_elev_sd_max = np.std(st_elev_per_max, axis=0, ddof=1)
        st_axial_sd_max = np.std(st_axial_per_max, axis=0, ddof=1)
        st_poe_sd_max = np.std(st_poe_per_max, axis=0, ddof=1)
        gh_elev_sd_max = np.std(gh_elev_per_max, axis=0, ddof=1)
        gh_axial_sd_max = np.std(gh_axial_per_max, axis=0, ddof=1)
        gh_poe_sd_max = np.std(gh_poe_per_max, axis=0, ddof=1)

        # plot endpoints
        axs[cur_row, 0].errorbar(max_pos - 3, st_elev_mean_max, yerr=st_elev_sd_max,
                                 color=color_map.colors[0], marker=markers[0], capsize=3)
        axs[cur_row, 0].errorbar(max_pos, st_axial_mean_max, yerr=st_axial_sd_max,
                                 color=color_map.colors[1], marker=markers[0], capsize=3)
        axs[cur_row, 0].errorbar(max_pos + 3, st_poe_mean_max, yerr=st_poe_sd_max,
                                 color=color_map.colors[2], marker=markers[0], capsize=3)

        axs[cur_row, 1].errorbar(max_pos - 3, gh_elev_mean_max, yerr=gh_elev_sd_max,
                                 color=color_map.colors[0], marker=markers[1], capsize=3)
        axs[cur_row, 1].errorbar(max_pos, gh_axial_mean_max, yerr=gh_axial_sd_max,
                                 color=color_map.colors[1], marker=markers[1], capsize=3)
        axs[cur_row, 1].errorbar(max_pos + 3, gh_poe_mean_max, yerr=gh_poe_sd_max,
                                 color=color_map.colors[2], marker=markers[1], capsize=3)

        # plot title
        axs[cur_row, 0].set_title('ST Joint', y=0.88)
        axs[cur_row, 1].set_title('GH Joint', y=0.88)

        # print percentages at maximum
        print(activity)
        print('ST Elevation {:.2f}'.format(st_elev_mean_max))
        print('ST Axial Rotation {:.2f}'.format(st_axial_mean_max))
        print('ST PoE {:.2f}'.format(st_poe_mean_max))

        print('GH Elevation {:.2f}'.format(gh_elev_mean_max))
        print('GH Axial Rotation {:.2f}'.format(gh_axial_mean_max))
        print('GH PoE {:.2f}'.format(gh_poe_mean_max))

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

    for row in axs:
        for ax in row:
            ax.set_xticks(x_ticks)
            tick_labels = [str(i) for i in x_ticks]
            tick_labels[-1] = 'Max'
            ax.set_xticklabels(tick_labels)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.03, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
