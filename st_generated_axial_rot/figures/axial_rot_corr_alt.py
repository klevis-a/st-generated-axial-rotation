"""Correlation of ST-generated Axial Rotation with GH PoE

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
    from scipy.stats import linregress
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

    config_dir = Path(mod_arg_parser('Correlation of ST-generated Axial Rotation with Euler/Cardan angles',
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
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}
    markers = {'ca': 'o', 'sa': 's', 'fe': '^'}
    colors = {'ca': 1, 'sa': 2, 'fe': 0}

    x_elev = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 100 / 25.4), dpi=params.dpi)
    axs_elev = fig.subplots(1, 3)

    y_labels = ['GH PoE (deg)', 'ST-contributed Axial Rotation (deg)', 'ST-contributed Axial Rotation (deg)']
    for i in range(3):
        style_axes(axs_elev[i], 'Mean GH PoE (deg)' if i == 2 else 'Humerothoracic Elevation (deg)', y_labels[i])
        if i == 0:
            axs_elev[i].set_ylim(-40, 38)
        else:
            axs_elev[i].set_ylim(-35, 30)
    # plot
    max_pos = 140
    leg_mean = []
    poes = []
    st_axials = []
    r_loc = [(-10, -25), (30, 10), (10, -5)]
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

        color = color_map.colors[colors[activity.lower()]]
        marker = markers[activity.lower()]
        # plot mean and SD
        poe_ln = mean_sd_plot(axs_elev[0], x_elev, poe_mean, poe_sd,
                              dict(color=color, alpha=0.25, hatch='oo' if activity == 'CA' else None),
                              dict(color=color, marker=marker, markevery=20, ms=4))
        st_axial_ln = mean_sd_plot(axs_elev[1], x_elev, st_axial_mean, st_axial_sd,
                                   dict(color=color, alpha=0.25, hatch='oo' if activity == 'CA' else None),
                                   dict(color=color, marker=marker, markevery=20, ms=4))

        # at maximum
        traj_poe_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'euler.gh_phadke', 1, None]), axis=0)
        traj_st_axial_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0)

        poe_mean_max = np.rad2deg(np.mean(traj_poe_max, axis=0))
        st_axial_mean_max = np.rad2deg(np.mean(traj_st_axial_max, axis=0))
        poe_sd_max = np.rad2deg(np.std(traj_poe_max, axis=0, ddof=1))
        st_axial_sd_max = np.rad2deg(np.std(traj_st_axial_max, axis=0, ddof=1))

        axs_elev[0].errorbar(max_pos + 3 * (act_idx - 1), poe_mean_max, yerr=poe_sd_max,
                             color=color, marker=marker, capsize=3, ms=4)
        axs_elev[1].errorbar(max_pos + 3 * (act_idx - 1), st_axial_mean_max, yerr=st_axial_sd_max,
                             color=color, marker=marker, capsize=3, ms=4)

        # scatter
        poe_mean_by_subj = np.rad2deg(np.mean(traj_poe, axis=1))
        # at 90 deg of HT elevation
        # poe_mean_by_subj = np.rad2deg(traj_poe[:, 261])
        # at maximum
        # max_idx = np.argmax(np.abs(traj_poe), axis=1)
        # poe_mean_by_subj = np.rad2deg(traj_poe[np.arange(max_idx.size), max_idx])
        st_axial_max = np.rad2deg(traj_st_axial_max)
        # axs[2].scatter(poe_mean_by_subj, np.rad2deg(traj_st_axial[:, -1]), c=color_map.colors[act_idx])
        axs_elev[2].scatter(poe_mean_by_subj, st_axial_max, c=color, s=12, marker=marker)

        slope, intercept, r_value, p_value, _ = linregress(poe_mean_by_subj, st_axial_max)

        # plot regression line
        xmin = np.min(poe_mean_by_subj)
        xmax = np.max(poe_mean_by_subj)
        xrange = np.arange(xmin, xmax + 0.1, 0.1)
        axs_elev[2].plot(xrange, slope * xrange + intercept, c=color, lw=2)

        print(activity)
        print('GH PoE Mean Overall: {:.2f}'.format(np.rad2deg(np.mean(traj_poe))))
        print('GH PoE Mean Min {:.2f}:'.format(np.min(poe_mean_by_subj)))
        print('GH PoE Mean Max {:.2f}:'.format(np.max(poe_mean_by_subj)))
        print('ST-Generated Mean at Max: {:.2f}'.format(st_axial_mean_max))
        print('ST_Generated Min at Max {:.2f}:'.format(np.min(st_axial_max)))
        print('ST_Generated Max at Max {:.2f}:'.format(np.max(st_axial_max)))
        print('r-value: {:.2f}'.format(r_value))
        print('p-value: {:.5f}'.format(p_value))
        print('Slope: {:.5f}'.format(slope))
        axs_elev[2].text(r_loc[act_idx][0], r_loc[act_idx][1], 'R={:.2f}'.format(r_value), color=color,
                         fontweight='bold')

        poes.append(poe_mean_by_subj)
        st_axials.append(st_axial_max)

        leg_mean.append(poe_ln[0])

    # overall regression
    all_poes = np.concatenate(poes)
    all_st_axials = np.concatenate(st_axials)
    slope, intercept, r_value, p_value, _ = linregress(all_poes, all_st_axials)
    xmin = np.min(all_poes)
    xmax = np.max(all_poes)
    xrange = np.arange(xmin, xmax + 0.1, 0.1)
    axs_elev[2].plot(xrange, slope * xrange + intercept, '--', c='k', lw=3)
    print('Elevation Combined')
    print('GH PoE Mean Overall: {:.2f}'.format(np.mean(all_poes)))
    print('GH PoE Mean Min {:.2f}:'.format(np.min(all_poes)))
    print('GH PoE Mean Max {:.2f}:'.format(np.max(all_poes)))
    print('ST-Generated Mean at Max: {:.2f}'.format(np.mean(all_st_axials)))
    print('ST_Generated Min at Max {:.2f}:'.format(np.min(all_st_axials)))
    print('ST_Generated Max at Max {:.2f}:'.format(np.max(all_st_axials)))
    print('r-value: {:.2f}'.format(r_value))
    print('p-value: {:.5f}'.format(p_value))
    print('Slope: {:.5f}'.format(slope))
    axs_elev[2].text(-15, 0, 'R={:.2f}'.format(r_value), color='k', fontweight='bold', fontsize=11)

    x_ticks = np.array([20, 40, 60, 80, 100, 120, 140])
    for i in range(2):
        axs_elev[i].set_xticks(x_ticks)
        tick_labels = [str(i) for i in x_ticks]
        tick_labels[-1] = 'Max'
        axs_elev[i].set_xticklabels(tick_labels)

    # labels for elevation trials
    leg_left = fig.legend([leg_mean[i] for i in [0, 2, 1]], ['CA', 'SA', 'FE'], loc='upper left',
                          bbox_to_anchor=(0, 1.01), ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          labelspacing=0.3, borderpad=0.2)
    axs_elev[0].set_title('GH PoE', y=0.925)
    axs_elev[1].set_title('ST-contributed\nAxial Rotation', y=0.88)
    axs_elev[2].set_title('ST-contributed\nAxial Rotation\nvs GH PoE', y=0.86)

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0)
    fig.suptitle('ST-contributed Axial Rotation versus GH PoE', x=0.5, y=0.99,
                 fontweight='bold')
    plt.subplots_adjust(top=0.93)

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
