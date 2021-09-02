"""Correlation of ST-generated Axial Rotation between Activities

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
    from st_generated_axial_rot.common.analysis_utils import prepare_db, sub_rot_at_max_elev
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Correlation of ST-generated Axial Rotation between Activities',
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
    db_elev_equal = db_elev.loc[~db_elev['Trial_Name'].str.contains('U35_010|U35_002')].copy()
    db_er_endpts_equal = db_er_endpts.loc[~db_er_endpts['Trial_Name'].str.contains('U35_010|U35_002|O45_003')].copy()

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
        style_axes(axs_elev[i], 'ST Axial Rotation (Deg)' if i == 2 else None, 'ST Axial Rotation (Deg)')

    for i in range(2):
        style_axes(axs_er[i], 'ST Axial Rotation (Deg)' if i == 1 else None, 'ST Axial Rotation (Deg)')

    ca_df = db_elev_equal.loc[db['Trial_Name'].str.contains('_CA_')].copy()
    sa_df = db_elev_equal.loc[db['Trial_Name'].str.contains('_SA_')].copy()
    fe_df = db_elev_equal.loc[db['Trial_Name'].str.contains('_FE_')].copy()

    traj_st_axial_ca = np.rad2deg(np.stack(ca_df['traj_interp'].apply(
        sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0))
    traj_st_axial_sa = np.rad2deg(np.stack(sa_df['traj_interp'].apply(
        sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0))
    traj_st_axial_fe = np.rad2deg(np.stack(fe_df['traj_interp'].apply(
        sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0))

    elev_pairs = [(traj_st_axial_ca, traj_st_axial_sa), (traj_st_axial_ca, traj_st_axial_fe),
                  (traj_st_axial_sa, traj_st_axial_fe)]
    elev_pair_names = ['SA vs CA', 'FE vs CA', 'FE vs SA']

    def plot_scatter(ax, x, y, r_loc):
        ax.scatter(x, y)
        xmin = np.min(x)
        xmax = np.max(x)
        xrange = np.arange(xmin, xmax + 0.1, 0.1)
        slope, intercept, r_value, p_value, _ = linregress(x, y)
        ax.plot(xrange, slope * xrange + intercept, c='k', lw=2)
        ax.text(r_loc[0], r_loc[1], 'R={:.2f}'.format(r_value), color='k', fontweight='bold')
        print('r-value: {:.2f}'.format(r_value))
        print('p-value: {:.5f}'.format(p_value))
        print('Slope: {:.5f}'.format(slope))

    r_loc = [(-32, -20), (-32, 11), (-27, 11)]
    for idx, elev_pair in enumerate(elev_pairs):
        print(elev_pair_names[idx])
        plot_scatter(axs_elev[idx], elev_pair[0], elev_pair[1], r_loc[idx])

    erar_df = db_er_endpts_equal.loc[db['Trial_Name'].str.contains('_ERaR_')].copy()
    era90_df = db_er_endpts_equal.loc[db['Trial_Name'].str.contains('_ERa90_')].copy()
    era90_df_vs_erar = era90_df.loc[~db_er_endpts['Trial_Name'].str.contains('U35_001')].copy()

    traj_st_axial_erar = np.rad2deg(np.stack(erar_df['st_contribs_interp'], axis=0)[:, -1, 2])
    traj_st_axial_era90 = np.rad2deg(np.stack(era90_df['st_contribs_interp'], axis=0)[:, -1, 2])
    traj_st_axial_era90_erar = np.rad2deg(np.stack(era90_df_vs_erar['st_contribs_interp'], axis=0)[:, -1, 2])

    er_pairs = [(traj_st_axial_era90_erar, traj_st_axial_erar), (traj_st_axial_ca, traj_st_axial_era90)]
    er_pair_names = ['ER-ADD vs ER-ABD', 'ER-ABD vs CA']
    r_loc_er = [(-30, -12), (-32, -22)]
    for idx, er_pair in enumerate(er_pairs):
        print(er_pair_names[idx])
        plot_scatter(axs_er[idx], er_pair[0], er_pair[1], r_loc_er[idx])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('Correlation of ST-contributed Axial Rotation between Activities', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)

    # add axes titles
    _, y0, _, h = axs_elev[0].get_position().bounds
    fig.text(0.25, y0 + h * 1.02, 'SA vs CA', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_elev[1].get_position().bounds
    fig.text(0.25, y0 + h * 1.02, 'FE vs CA', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_elev[2].get_position().bounds
    fig.text(0.25, y0 + h * 1.02, 'FE vs SA', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_er[0].get_position().bounds
    fig.text(0.75, y0 + h * 1.01, 'ER-ADD vs ER-ABD', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_er[1].get_position().bounds
    fig.text(0.77, y0 + h * 1.01, r'ER-ABD vs CA', ha='center', fontsize=11,
             fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
