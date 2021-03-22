"""How PoE affects ST axial rotation

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
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from scipy.stats import linregress
    from scipy.integrate import cumtrapz
    import numpy as np
    import matplotlib.pyplot as plt
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import (prepare_db, extract_sub_rot, extract_sub_rot_norm,
                                                              sub_rot_at_max_elev)
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('How PoE affects ST axial rotation',
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

    #%%
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 90 / 25.4))
    axs = fig.subplots(1, 3)

    for i in range(3):
        style_axes(axs[i], 'Humerothoracic Elevation (deg)', 'PoE (deg)' if i == 0 else 'ST Axial Rotation (deg)')

    # # set axes limits
    # ax_limits = [(20, 70), (-40, 50)]
    # for i in range(3):
    #     for j in range(2):
    #         axs[i, j].set_ylim(ax_limits[j][0], ax_limits[j][1])
    #         axs[i, j].yaxis.set_major_locator(ticker.MultipleLocator(10))
    #         style_axes(axs[i, j], 'Humerothoracic Elevation (Deg)' if i == 2 else None,
    #                    'ReProtraction (Deg)' if j == 0 else 'PoE (Deg)')

    # plot
    max_pos = 140
    leg_mean = []
    for act_idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        traj_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_phadke', 1]), axis=0)
        traj_st_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 2, 'up']), axis=0)

        # means and standard deviations
        poe_mean = np.rad2deg(np.mean(traj_poe, axis=0))
        st_axial_mean = np.rad2deg(np.mean(traj_st_axial, axis=0))
        poe_sd = np.rad2deg(np.std(traj_poe, axis=0, ddof=1))
        st_axial_sd = np.rad2deg(np.std(traj_st_axial, axis=0, ddof=1))

        st_axial_med = np.rad2deg(np.median(traj_st_axial, axis=0))
        st_axial_25 = np.rad2deg(np.quantile(traj_st_axial, 0.25, axis=0))
        st_axial_75 = np.rad2deg(np.quantile(traj_st_axial, 0.75, axis=0))

        # plot mean and SD
        poe_ln = mean_sd_plot(axs[0], x, poe_mean, poe_sd,
                              dict(color=color_map.colors[act_idx], alpha=0.25),
                              dict(color=color_map.colors[act_idx], marker=markers[0], markevery=20))
        st_axial_ln = mean_sd_plot(axs[1], x, st_axial_mean, st_axial_sd,
                                   dict(color=color_map.colors[act_idx], alpha=0.25),
                                   dict(color=color_map.colors[act_idx], marker=markers[1], markevery=20))

        # at maximum
        traj_poe_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_phadke', 1, None]), axis=0)
        traj_st_axial_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0)

        poe_mean_max = np.rad2deg(np.mean(traj_poe_max, axis=0))
        st_axial_mean_max = np.rad2deg(np.mean(traj_st_axial_max, axis=0))
        poe_sd_max = np.rad2deg(np.std(traj_poe_max, axis=0, ddof=1))
        st_axial_sd_max = np.rad2deg(np.std(traj_st_axial_max, axis=0, ddof=1))

        axs[0].errorbar(max_pos + 3 * (act_idx - 1), poe_mean_max, yerr=poe_sd_max,
                        color=color_map.colors[act_idx], marker=markers[0], capsize=3)
        axs[1].errorbar(max_pos + 3 * (act_idx - 1), st_axial_mean_max, yerr=st_axial_sd_max,
                        color=color_map.colors[act_idx], marker=markers[1], capsize=3)

        # scatter
        traj_st_upward = -np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'euler.st_isb', 1, 'up']), axis=0)
        poe_int = cumtrapz(np.rad2deg(traj_poe), x=np.rad2deg(traj_st_upward), axis=1, initial=0)
        poe_mean_by_subj = np.rad2deg(np.max(traj_poe, axis=1))
        st_axial_max = np.rad2deg(traj_st_axial_max)
        axs[2].scatter(poe_int[:, -1], np.rad2deg(traj_st_axial[:, -1]), c=color_map.colors[act_idx])

        slope, intercept, r_value, p_value, std_err = linregress(poe_int[:, -1], np.rad2deg(traj_st_axial[:, -1]))

        print(activity)
        print('r-value: {:.2f}'.format(r_value))

        if act_idx == 0:
            leg_mean.append(poe_ln[0])
            leg_mean.append(st_axial_ln[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0)
    fig.suptitle('PoE Comparison', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)

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
    axs[0].set_xticks(x_ticks)
    axs[1].set_xticks(x_ticks)

    # add arrows indicating direction
    # axs[0, 0].arrow(35, -15, 0, -15, length_includes_head=True, head_width=2, head_length=2)
    # axs[0, 0].text(23, -15, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles

    make_interactive()

    plt.show()
