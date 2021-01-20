"""Compare HT, ST, and GH true axial rotation and measure against zero.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
use_ac: Whether to use the AC or GC landmark when building the scapula CS.
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
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, spm_plot,
                                                          style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare HT, ST, and GH true axial rotation and measure against zero',
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
    use_ac = bool(distutils.util.strtobool(params.use_ac))

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, use_ac, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])

    #%%
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig_axial_hum = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_axial = fig_axial_hum.subplots(3, 2)
    fig_axial_hum_norm = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_axial_norm = fig_axial_hum_norm.subplots(3, 2)

    # style axes, add x and y labels
    style_axes(axs_axial[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_axial[0, 1], None, 'SPM{t}')
    style_axes(axs_axial[1, 1], None, 'SPM{t}')
    style_axes(axs_axial[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

    style_axes(axs_axial_norm[0, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial_norm[1, 0], None, 'Axial Rotation (Deg)')
    style_axes(axs_axial_norm[2, 0], 'Humerothoracic Elevation (Deg)', 'Axial Rotation (Deg)')
    style_axes(axs_axial_norm[0, 1], None, 'SPM{t}')
    style_axes(axs_axial_norm[1, 1], None, 'SPM{t}')
    style_axes(axs_axial_norm[2, 1], 'Humerothoracic Elevation (Deg)', 'SPM{t}')

    # plot
    leg_patch_mean = []
    leg_patch_mean_norm = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_true_ht = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_true_gh = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_true_st = all_traj_true_ht - all_traj_true_gh
        all_traj_true_ht_norm = all_traj_true_ht - all_traj_true_ht[:, 0][..., np.newaxis]
        all_traj_true_gh_norm = all_traj_true_gh - all_traj_true_gh[:, 0][..., np.newaxis]
        all_traj_true_st_norm = all_traj_true_st - all_traj_true_st[:, 0][..., np.newaxis]

        # means and standard deviations
        true_mean_ht = np.rad2deg(np.mean(all_traj_true_ht, axis=0))
        true_mean_gh = np.rad2deg(np.mean(all_traj_true_gh, axis=0))
        true_mean_st = np.rad2deg(np.mean(all_traj_true_st, axis=0))
        true_sd_ht = np.rad2deg(np.std(all_traj_true_ht, ddof=1, axis=0))
        true_sd_gh = np.rad2deg(np.std(all_traj_true_gh, ddof=1, axis=0))
        true_sd_st = np.rad2deg(np.std(all_traj_true_st, ddof=1, axis=0))

        true_mean_ht_norm = np.rad2deg(np.mean(all_traj_true_ht_norm, axis=0))
        true_mean_gh_norm = np.rad2deg(np.mean(all_traj_true_gh_norm, axis=0))
        true_mean_st_norm = np.rad2deg(np.mean(all_traj_true_st_norm, axis=0))
        true_sd_ht_norm = np.rad2deg(np.std(all_traj_true_ht_norm, ddof=1, axis=0))
        true_sd_gh_norm = np.rad2deg(np.std(all_traj_true_gh_norm, ddof=1, axis=0))
        true_sd_st_norm = np.rad2deg(np.std(all_traj_true_st_norm, ddof=1, axis=0))

        # spm
        ht_zero = spm1d.stats.ttest(all_traj_true_ht, 0).inference(alpha, two_tailed=True)
        gh_zero = spm1d.stats.ttest(all_traj_true_gh, 0).inference(alpha, two_tailed=True)
        st_zero = spm1d.stats.ttest(all_traj_true_st, 0).inference(alpha, two_tailed=True)
        ht_zero_norm = spm1d.stats.ttest(all_traj_true_ht_norm[:, 1:], 0).inference(alpha, two_tailed=True)
        gh_zero_norm = spm1d.stats.ttest(all_traj_true_gh_norm[:, 1:], 0).inference(alpha, two_tailed=True)
        st_zero_norm = spm1d.stats.ttest(all_traj_true_st_norm[:, 1:], 0).inference(alpha, two_tailed=True)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        true_ht_ln = mean_sd_plot(axs_axial[cur_row, 0], x, true_mean_ht, true_sd_ht,
                                  dict(color=color_map.colors[2], alpha=0.25),
                                  dict(color=color_map.colors[2], marker=markers[0], markevery=20))
        true_gh_ln = mean_sd_plot(axs_axial[cur_row, 0], x, true_mean_gh, true_sd_gh,
                                  dict(color=color_map.colors[0], alpha=0.25),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        true_st_ln = mean_sd_plot(axs_axial[cur_row, 0], x, true_mean_st, true_sd_st,
                                  dict(color=color_map.colors[1], alpha=0.25),
                                  dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        true_ht_ln_norm = mean_sd_plot(axs_axial_norm[cur_row, 0], x, true_mean_ht_norm, true_sd_ht_norm,
                                       dict(color=color_map.colors[2], alpha=0.25),
                                       dict(color=color_map.colors[2], marker=markers[0], markevery=20))
        true_gh_ln_norm = mean_sd_plot(axs_axial_norm[cur_row, 0], x, true_mean_gh_norm, true_sd_gh_norm,
                                       dict(color=color_map.colors[0], alpha=0.25),
                                       dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        true_st_ln_norm = mean_sd_plot(axs_axial_norm[cur_row, 0], x, true_mean_st_norm, true_sd_st_norm,
                                       dict(color=color_map.colors[1], alpha=0.25),
                                       dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        # plot spm
        ht_t_ln = spm_plot(axs_axial[cur_row, 1], x, ht_zero, dict(color=color_map.colors[2], alpha=0.25),
                           dict(color=color_map.colors[2]))
        gh_t_ln = spm_plot(axs_axial[cur_row, 1], x, gh_zero, dict(color=color_map.colors[0], alpha=0.25),
                           dict(color=color_map.colors[0]))
        st_t_ln = spm_plot(axs_axial[cur_row, 1], x, st_zero, dict(color=color_map.colors[1], alpha=0.25),
                           dict(color=color_map.colors[1]))
        ht_norm_t_ln = spm_plot(axs_axial_norm[cur_row, 1], x[1:], ht_zero_norm,
                                dict(color=color_map.colors[2], alpha=0.25), dict(color=color_map.colors[2]))
        gh_norm_t_ln = spm_plot(axs_axial_norm[cur_row, 1], x[1:], gh_zero_norm,
                                dict(color=color_map.colors[0], alpha=0.25), dict(color=color_map.colors[0]))
        st_norm_t_ln = spm_plot(axs_axial_norm[cur_row, 1], x[1:], st_zero_norm,
                                dict(color=color_map.colors[1], alpha=0.25), dict(color=color_map.colors[1]))

        if idx == 0:
            leg_patch_mean.append(true_ht_ln[0])
            leg_patch_mean.append(true_st_ln[0])
            leg_patch_mean.append(true_gh_ln[0])
            leg_patch_mean_norm.append(true_ht_ln_norm[0])
            leg_patch_mean_norm.append(true_gh_ln_norm[0])
            leg_patch_mean_norm.append(true_st_ln_norm[0])

    # figure title and legend
    plt.figure(fig_axial_hum.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    plt.figure(fig_axial_hum_norm.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_axial_hum.suptitle('HT, GH, and ST True Axial Rotation Comparison', x=0.4, y=0.99, fontweight='bold')
    fig_axial_hum_norm.suptitle('HT, GH, and ST True Axial Rotation Comparison Norm Start', x=0.4, y=0.99,
                                fontweight='bold')
    plt.figure(fig_axial_hum.number)
    plt.subplots_adjust(top=0.93)
    plt.figure(fig_axial_hum_norm.number)
    plt.subplots_adjust(top=0.93)
    fig_axial_hum.legend(
        leg_patch_mean, ['HT', 'ST', 'GH'],
        loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75)
    fig_axial_hum_norm.legend(
        leg_patch_mean, ['HT', 'ST', 'GH'],
        loc='upper right', bbox_to_anchor=(1.01, 1.01), ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75)

    # add axes titles
    _, y0, _, h = axs_axial[0, 0].get_position().bounds
    fig_axial_hum.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[1, 0].get_position().bounds
    fig_axial_hum.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[2, 0].get_position().bounds
    fig_axial_hum.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial_norm[0, 0].get_position().bounds
    fig_axial_hum_norm.text(0.5, y0 + h * 1.05, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial_norm[1, 0].get_position().bounds
    fig_axial_hum_norm.text(0.5, y0 + h * 1.05, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial_norm[2, 0].get_position().bounds
    fig_axial_hum_norm.text(0.5, y0 + h * 1.05, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    plt.figure(fig_axial_hum.number)
    make_interactive()
    plt.figure(fig_axial_hum_norm.number)
    make_interactive()
    plt.show()
