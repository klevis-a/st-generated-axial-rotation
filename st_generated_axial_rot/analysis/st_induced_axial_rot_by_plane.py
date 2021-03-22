"""Compare ST induced HT axial rotation between planes of elevation.

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
    import numpy as np
    import matplotlib.pyplot as plt
    import spm1d
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, spm_plot_alpha,
                                                          style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import (prepare_db, extract_sub_rot_norm,
                                                              st_induced_axial_rot_fha, add_st_induced,
                                                              subj_name_to_number)
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("ST induced HT axial rotation comparison by elevation plane",
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
    db_elev_equal = db_elev.loc[~db_elev['Trial_Name'].str.contains('U35_010')].copy()
    db_elev['traj_interp'].apply(add_st_induced, args=[st_induced_axial_rot_fha])

    #%%
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_plane = plt.figure(figsize=(190 / 25.4, 230 / 25.4))
    axs_plane = fig_plane.subplots(4, 2)

    # style axes, add x and y labels
    style_axes(axs_plane[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_plane[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_plane[2, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_plane[3, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_plane[0, 1], None, 'SPM{F}')
    style_axes(axs_plane[1, 1], None, 'SPM{F}')
    style_axes(axs_plane[2, 1], None, 'SPM{F}')
    style_axes(axs_plane[3, 1], 'Humerothoracic Elevation (Deg)', 'SPM{F}')

    all_traj_induced = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 3, 'up']), axis=0)
    all_traj_induced_x = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 0, 'up']), axis=0)
    all_traj_induced_y = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 1, 'up']), axis=0)
    all_traj_induced_z = np.stack(db_elev_equal['traj_interp'].apply(
        extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 2, 'up']), axis=0)

    group_rm = (db_elev_equal['Activity'].map({'CA': 1, 'SA': 2, 'FE': 3})).to_numpy(dtype=np.int)
    subj_rm = (db_elev_equal['Subject_Short'].map(subj_name_to_number)).to_numpy()

    spm_one_way_rm_induced = spm1d.stats.anova1rm(all_traj_induced, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_induced_x = spm1d.stats.anova1rm(all_traj_induced_x, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_induced_y = spm1d.stats.anova1rm(all_traj_induced_y, group_rm, subj_rm).inference(alpha=0.05)
    spm_one_way_rm_induced_z = spm1d.stats.anova1rm(all_traj_induced_z, group_rm, subj_rm).inference(alpha=0.05)

    shaded_spm_rm = dict(color=color_map.colors[6], alpha=0.25)
    line_spm_rm = dict(color=color_map.colors[7])
    one_way_rm_ln_induced, alpha_ln_induced = spm_plot_alpha(axs_plane[0, 1], x, spm_one_way_rm_induced,
                                                             shaded_spm_rm, line_spm_rm)
    one_way_rm_ln_induced_x, alpha_ln_induced_x = spm_plot_alpha(axs_plane[1, 1], x, spm_one_way_rm_induced_x,
                                                                 shaded_spm_rm, line_spm_rm)
    one_way_rm_ln_induced_y, alpha_ln_induced_y = spm_plot_alpha(axs_plane[2, 1], x, spm_one_way_rm_induced_y,
                                                                 shaded_spm_rm, line_spm_rm)
    one_way_rm_ln_induced_z, alpha_ln_induced_z = spm_plot_alpha(axs_plane[3, 1], x, spm_one_way_rm_induced_z,
                                                                 shaded_spm_rm, line_spm_rm)

    # print('True Axial')
    # print(extract_sig(spm_one_way_rm_true_start, x))
    # print('ISB')
    # print(extract_sig(spm_one_way_rm_isb_start, x))
    # print('Phadke')
    # print(extract_sig(spm_one_way_rm_phadke_start, x))

    mean_lns_start = []
    activities_start = []
    for idx, (activity, activity_df) in enumerate(db_elev_equal.groupby('Activity', observed=True)):
        act_all_traj_induced = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 3, 'up']), axis=0)
        act_all_traj_induced_x = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 0, 'up']), axis=0)
        act_all_traj_induced_y = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 1, 'up']), axis=0)
        act_all_traj_induced_z = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 2, 'up']), axis=0)

        # means and standard deviations
        induced_mean = np.rad2deg(np.mean(act_all_traj_induced, axis=0))
        induced_mean_x = np.rad2deg(np.mean(act_all_traj_induced_x, axis=0))
        induced_mean_y = np.rad2deg(np.mean(act_all_traj_induced_y, axis=0))
        induced_mean_z = np.rad2deg(np.mean(act_all_traj_induced_z, axis=0))

        induced_sd = np.rad2deg(np.std(act_all_traj_induced, ddof=1, axis=0))
        induced_sd_x = np.rad2deg(np.std(act_all_traj_induced_x, ddof=1, axis=0))
        induced_sd_y = np.rad2deg(np.std(act_all_traj_induced_y, ddof=1, axis=0))
        induced_sd_z = np.rad2deg(np.std(act_all_traj_induced_z, ddof=1, axis=0))

        # plot mean +- sd
        shaded = dict(color=color_map.colors[idx], alpha=0.2)
        if activity == 'CA':
            shaded['hatch'] = 'oo'
        if activity == 'SA':
            shaded['alpha'] = 0.28
        if activity == 'FE':
            shaded['alpha'] = 0.3

        line = dict(color=color_map.colors[idx], marker=markers[idx], markevery=20)

        induced_ln = mean_sd_plot(axs_plane[0, 0], x, induced_mean, induced_sd, shaded, line)
        induced_x_ln = mean_sd_plot(axs_plane[1, 0], x, induced_mean_x, induced_sd_x, shaded, line)
        induced_y_ln = mean_sd_plot(axs_plane[2, 0], x, induced_mean_y, induced_sd_y, shaded, line)
        induced_z_ln = mean_sd_plot(axs_plane[3, 0], x, induced_mean_z, induced_sd_z, shaded, line)

        mean_lns_start.append(induced_ln[0])
        activities_start.append(activity)

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig_plane.suptitle('ST-induced HT Axial Rotation Comparison by Plane', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig_plane.legend(mean_lns_start, activities_start, loc='upper left', bbox_to_anchor=(0, 1),
                                ncol=2, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, borderpad=0.2,
                                labelspacing=0.4)
    leg_right = fig_plane.legend([one_way_rm_ln_induced[0], alpha_ln_induced],
                                 ['CA = SA = FE', '$\\alpha=0.05$'], loc='upper right', borderpad=0.2,
                                 bbox_to_anchor=(1, 1), handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                                 labelspacing=0.4)

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
    for row in axs_plane:
        for ax in row:
            ax.set_xticks(x_ticks)

    # add arrows indicating direction
    axs_plane[0, 0].arrow(35, -10, 0, -10, length_includes_head=True, head_width=2, head_length=2)
    axs_plane[0, 0].text(23, -7, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs_plane[0, 0].get_position().bounds
    fig_plane.text(0.5, y0 + h * 1.02, 'Total Induced HT Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_plane[1, 0].get_position().bounds
    fig_plane.text(0.5, y0 + h * 1.02, 'HT Axial Rotation Induced from Lateral Rotation', ha='center', fontsize=11,
                   fontweight='bold')

    _, y0, _, h = axs_plane[2, 0].get_position().bounds
    fig_plane.text(0.5, y0 + h * 1.02, 'HT Axial Rotation Induced from Protraction', ha='center', fontsize=11,
                   fontweight='bold')

    _, y0, _, h = axs_plane[3, 0].get_position().bounds
    fig_plane.text(0.5, y0 + h * 1.02, 'HT Axial Rotation Induced from Tilt', ha='center', fontsize=11,
                   fontweight='bold')
    make_interactive()

    # normality tests
    spm_one_way_rm_induced_norm = spm1d.stats.normality.sw.anova1rm(all_traj_induced, group_rm, subj_rm)
    spm_one_way_rm_induced_x_norm = spm1d.stats.normality.sw.anova1rm(all_traj_induced_x, group_rm, subj_rm)
    spm_one_way_rm_induced_y_norm = spm1d.stats.normality.sw.anova1rm(all_traj_induced_y, group_rm, subj_rm)
    spm_one_way_rm_induced_z_norm = spm1d.stats.normality.sw.anova1rm(all_traj_induced_z, group_rm, subj_rm)

    fig_norm = plt.figure()
    ax_norm_start = fig_norm.subplots()
    ax_norm_start.axhline(0.05, ls='--', color='grey')
    norm_rm_induced_ln = ax_norm_start.plot(x, spm_one_way_rm_induced_norm[1], color=color_map.colors[0], ls='--')
    norm_rm_induced_x_ln = ax_norm_start.plot(x, spm_one_way_rm_induced_x_norm[1], color=color_map.colors[1], ls='--')
    norm_rm_induced_y_ln = ax_norm_start.plot(x, spm_one_way_rm_induced_y_norm[1], color=color_map.colors[2], ls='--')
    norm_rm_induced_z_ln = ax_norm_start.plot(x, spm_one_way_rm_induced_z_norm[1], color=color_map.colors[3], ls='--')

    fig_norm.legend([norm_rm_induced_ln[0], norm_rm_induced_x_ln[0], norm_rm_induced_y_ln[0], norm_rm_induced_z_ln[0]],
                    ['Total', 'MedLat', 'RePro', 'Tilt'], loc='upper right', ncol=2,
                    handlelength=1.5, handletextpad=0.5, columnspacing=0.75)
    style_axes(ax_norm_start, 'Humerothoracic Elevation (Deg)', 'p-value')
    plt.tight_layout()
    fig_norm.suptitle('Normality tests')
    plt.show()
