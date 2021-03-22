"""Correlation between ST-generated and GH Axial Rotation

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
    import numpy as np
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Correlation between ST-generated and GH Axial Rotation',
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
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    gs = fig.add_gridspec(6, 2)
    axs_elev = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[2:4, 0]), fig.add_subplot(gs[4:6, 0])]
    axs_er = [fig.add_subplot(gs[:3, 1]), fig.add_subplot(gs[3:6, 1])]

    for i in range(3):
        style_axes(axs_elev[i], 'GH Axial Rotation (Deg)' if i == 2 else None, 'ST Axial Rotation (Deg)')

    for i in range(2):
        style_axes(axs_er[i], 'GH Axial Rotation (Deg)' if i == 1 else None, 'ST Axial Rotation (Deg)')

    # plot elevation
    leg_elev_mean = []
    r_loc = [(10, -19), (20, -12), (19, 16)]
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        traj_st = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 2, 'up']), axis=0)
        traj_gh = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 2, 'up']), axis=0)
        traj_st_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0)
        traj_gh_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'contribs', 2, 'up']), axis=0)

        # scatter plot
        cur_row = act_row[activity.lower()]
        traj_gh_max_deg = np.rad2deg(traj_gh_max)
        traj_st_max_deg = np.rad2deg(traj_st_max)
        axs_elev[cur_row].scatter(traj_gh_max_deg, traj_st_max_deg)
        slope, intercept, r_value, p_value, _ = linregress(traj_gh_max_deg, traj_st_max_deg)

        # plot regression line
        xmin = np.min(traj_gh_max_deg)
        xmax = np.max(traj_gh_max_deg)
        xrange = np.arange(xmin, xmax + 0.1, 0.1)
        axs_elev[cur_row].plot(xrange, slope * xrange + intercept, c='k', lw=2)
        axs_elev[cur_row].text(r_loc[cur_row][0], r_loc[cur_row][1], 'R={:.2f}'.format(r_value), color='k',
                               fontweight='bold')
        print(activity)
        print('r-value: {:.2f}'.format(r_value))
        print('p-value: {:.5f}'.format(p_value))
        print('Slope: {:.5f}'.format(slope))

    # plot external rotation
    r_loc_er = [(-90, -11), (-70, -19)]
    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        traj_st = np.stack(activity_df['st_contribs_interp'], axis=0)[:, :, 2]
        traj_gh = np.stack(activity_df['gh_contribs_interp'], axis=0)[:, :, 2]

        # scatter plot
        traj_gh_deg = np.rad2deg(traj_gh[:, -1])
        traj_st_deg = np.rad2deg(traj_st[:, -1])
        axs_er[idx_act].scatter(traj_gh_deg, traj_st_deg)
        slope, intercept, r_value, p_value, _ = linregress(traj_gh_deg, traj_st_deg)

        # plot regression line
        xmin = np.min(traj_gh_deg)
        xmax = np.max(traj_gh_deg)
        xrange = np.arange(xmin, xmax + 0.1, 0.1)
        axs_er[idx_act].plot(xrange, slope * xrange + intercept, c='k', lw=2)
        axs_er[idx_act].text(r_loc_er[idx_act][0], r_loc_er[idx_act][1], 'R={:.2f}'.format(r_value), color='k',
                             fontweight='bold')

        print(activity)
        print('r-value: {:.2f}'.format(r_value))
        print('p-value: {:.5f}'.format(p_value))
        print('Slope: {:.5f}'.format(slope))

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('Correlation between ST-generated and GH Axial Rotation', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)

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
