"""Compare ST induced HT axial rotation against the null hypothesis of zero

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
                                                          HandlerTupleVertical, extract_sig, style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import (prepare_db, extract_sub_rot_norm,
                                                              st_induced_axial_rot_fha, add_st_induced)
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('ST induced HT axial rotation comparison against zero',
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
    db_elev['traj_interp'].apply(add_st_induced, args=[st_induced_axial_rot_fha])

    #%%
    if bool(distutils.util.strtobool(params.parametric)):
        spm_test = spm1d.stats.ttest
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest
        infer_params = {'force_iterations': True}

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_axial = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_axial = fig_axial.subplots(3, 2)

    fig_norm = plt.figure(figsize=(120 / 25.4, 190 / 25.4))
    ax_norm = fig_norm.subplots(3, 1)
    for ax in ax_norm:
        ax.axhline(0.05, ls='--', color='grey')
        style_axes(ax, 'Humerothoracic Elevation (Deg)', 'p-value')

    # style axes, add x and y labels
    style_axes(axs_axial[0, 0], None, 'Induced Axial Rotation (Deg)')
    style_axes(axs_axial[1, 0], None, 'Induced Axial Rotation (Deg)')
    style_axes(axs_axial[2, 0], 'Humerothoracic Elevation (Deg)', 'Induced Axial Rotation (Deg)')
    style_axes(axs_axial[0, 1], None, 'SPM{t}')
    style_axes(axs_axial[1, 1], None, 'SPM{t}')
    style_axes(axs_axial[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

    # plot
    leg_patch_mean = []
    leg_patch_t = []
    alpha_patch = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_induced = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 3, 'up']), axis=0)
        all_traj_induced_x = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 0, 'up']), axis=0)
        all_traj_induced_y = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 1, 'up']), axis=0)
        all_traj_induced_z = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 2, 'up']), axis=0)

        # means and standard deviations
        induced_mean = np.rad2deg(np.mean(all_traj_induced, axis=0))
        induced_mean_x = np.rad2deg(np.mean(all_traj_induced_x, axis=0))
        induced_mean_y = np.rad2deg(np.mean(all_traj_induced_y, axis=0))
        induced_mean_z = np.rad2deg(np.mean(all_traj_induced_z, axis=0))

        induced_sd = np.rad2deg(np.std(all_traj_induced, ddof=1, axis=0))
        induced_sd_x = np.rad2deg(np.std(all_traj_induced_x, ddof=1, axis=0))
        induced_sd_y = np.rad2deg(np.std(all_traj_induced_y, ddof=1, axis=0))
        induced_sd_z = np.rad2deg(np.std(all_traj_induced_z, ddof=1, axis=0))

        # spm
        induced_zero = spm_test(all_traj_induced, 0).inference(alpha, two_tailed=True, **infer_params)
        induced_x_zero = spm_test(all_traj_induced_x, 0).inference(alpha, two_tailed=True, **infer_params)
        induced_y_zero = spm_test(all_traj_induced_y, 0).inference(alpha, two_tailed=True, **infer_params)
        induced_z_zero = spm_test(all_traj_induced_z, 0).inference(alpha, two_tailed=True, **infer_params)

        print('Activity: {}'.format(activity))
        print('Total')
        print(extract_sig(induced_zero, x))
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean))))
        print('LatMed')
        print(extract_sig(induced_x_zero, x))
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean_x))))
        print('RePro')
        print(extract_sig(induced_y_zero, x))
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean_y))))
        print('Tilt')
        print(extract_sig(induced_z_zero, x))
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean_z))))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        induced_ln = mean_sd_plot(axs_axial[cur_row, 0], x, induced_mean, induced_sd,
                                  dict(color=color_map.colors[0], alpha=0.2, hatch='ooo'),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        induced_x_ln = mean_sd_plot(axs_axial[cur_row, 0], x, induced_mean_x, induced_sd_x,
                                    dict(color=color_map.colors[1], alpha=0.3),
                                    dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        induced_y_ln = mean_sd_plot(axs_axial[cur_row, 0], x, induced_mean_y, induced_sd_y,
                                    dict(color=color_map.colors[7], alpha=0.3),
                                    dict(color=color_map.colors[7], marker=markers[2], markevery=20))
        induced_z_ln = mean_sd_plot(axs_axial[cur_row, 0], x, induced_mean_z, induced_sd_z,
                                    dict(color=color_map.colors[3], alpha=0.2, hatch='xxx'),
                                    dict(color=color_map.colors[3], marker=markers[3], markevery=20))

        # plot spm
        induced_t_ln, induced_alpha = spm_plot_alpha(axs_axial[cur_row, 1], x, induced_zero,
                                                     dict(color=color_map.colors[0], alpha=0.25),
                                                     dict(color=color_map.colors[0]))
        induced_x_t_ln, induced_x_alpha = spm_plot_alpha(axs_axial[cur_row, 1], x, induced_x_zero,
                                                         dict(color=color_map.colors[1], alpha=0.25),
                                                         dict(color=color_map.colors[1]))
        induced_y_t_ln, induced_y_alpha = spm_plot_alpha(axs_axial[cur_row, 1], x, induced_y_zero,
                                                         dict(color=color_map.colors[2], alpha=0.25),
                                                         dict(color=color_map.colors[2]))
        induced_z_t_ln, induced_z_alpha = spm_plot_alpha(axs_axial[cur_row, 1], x, induced_z_zero,
                                                         dict(color=color_map.colors[3], alpha=0.25),
                                                         dict(color=color_map.colors[3]))

        # normality
        induced_norm = spm1d.stats.normality.sw.ttest(all_traj_induced)
        induced_x_norm = spm1d.stats.normality.sw.ttest(all_traj_induced_x)
        induced_y_norm = spm1d.stats.normality.sw.ttest(all_traj_induced_y)
        induced_z_norm = spm1d.stats.normality.sw.ttest(all_traj_induced_z)
        ax_norm[idx].plot(x, induced_norm[1], color=color_map.colors[0])
        ax_norm[idx].plot(x, induced_x_norm[1], color=color_map.colors[1])
        ax_norm[idx].plot(x, induced_y_norm[1], color=color_map.colors[2])
        ax_norm[idx].plot(x, induced_z_norm[1], color=color_map.colors[3])

        # empty line to make the legend look good
        empty_ln = axs_axial[cur_row, 1].plot(np.nan, color='none')

        if idx == 0:
            leg_patch_mean.append(induced_ln[0])
            leg_patch_mean.append(induced_x_ln[0])
            leg_patch_mean.append(induced_y_ln[0])
            leg_patch_mean.append(induced_z_ln[0])
            leg_patch_t.append(induced_t_ln[0])
            leg_patch_t.append(induced_x_t_ln[0])
            leg_patch_t.append(empty_ln[0])
            leg_patch_t.append(induced_y_t_ln[0])
            leg_patch_t.append(induced_z_t_ln[0])
            alpha_patch.append((induced_alpha, induced_x_alpha, induced_y_alpha, induced_z_alpha))

    # figure title and legend
    plt.figure(fig_axial.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_axial.suptitle('ST Induced HT Axial Rotation', y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig_axial.legend(leg_patch_mean, ['Total', 'MedLat', 'RePro', 'Tilt'], loc='upper left',
                                bbox_to_anchor=(0, 1), ncol=2, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                                labelspacing=0.3, borderpad=0.2)
    leg_right = fig_axial.legend(
        leg_patch_t + alpha_patch, ['Total = 0', 'MedLat = 0', '', 'Repro = 0', 'Tilt=0', '$\\alpha=0.05$'],
        loc='upper right', handler_map={tuple: HandlerTupleVertical(ndivide=None)}, bbox_to_anchor=(1, 1), ncol=2,
        handlelength=1.5, handletextpad=0.5, columnspacing=0.75, labelspacing=0.3, borderpad=0.2)

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
    for row in axs_axial:
        for ax in row:
            ax.set_xticks(x_ticks)

    # add arrows indicating direction
    # axs_axial[0, 0].arrow(35, -15, 0, -15, length_includes_head=True, head_width=2, head_length=2)
    # axs_axial[0, 0].text(23, -15, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs_axial[0, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[1, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[2, 0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    plt.figure(fig_norm.number)
    plt.tight_layout()
    fig_norm.suptitle('Normality tests')
    make_interactive()
    plt.show()
