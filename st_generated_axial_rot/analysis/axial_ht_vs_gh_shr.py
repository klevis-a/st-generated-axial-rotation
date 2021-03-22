"""Compare HT, ST, and GH true axial rotation and measure against zero. Also include Elevation and axial rotation SHR.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
dtheta_coarse: Incremental angle (deg) to use for coarse interpolation between minimum and maximum HT elevation analyzed.
min_elev: Minimum HT elevation angle (deg) utilized for analysis that encompasses all trials.
max_elev: Maximum HT elevation angle (deg) utilized for analysis that encompasses all trials.
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
parametric: Whether to use a parametric (true) or non-parametric statistical test (false).
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
                                                          HandlerTupleVertical, extract_sig, style_axes_right,
                                                          style_axes_add_right, style_axes)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare HT, ST, and GH true axial rotation and measure against zero with SHR',
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

    fig_axial_hum = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs_axial = fig_axial_hum.subplots(3, 2)

    # add twin axes for plotting SHR
    axs_axial_twin = np.empty((3,), dtype=object)
    axial_shr_dash = [1, 1, -1]
    yticks = [np.arange(0, 13, 2), np.arange(0, 16, 5), np.arange(-1, 3.1, 1)]
    ylim = [(0, 12), (0, 18), (-1.5, 3.5)]
    for i in range(3):
        axs_axial_twin[i] = axs_axial[i, 0].twinx()
        axs_axial_twin[i].set_yticks(yticks[i])
        style_axes_add_right(axs_axial_twin[i], 'Axial Rotation SHR')
        axs_axial_twin[i].axhline(axial_shr_dash[i], ls='--', color='grey')
        axs_axial_twin[i].set_ylim(ylim[i][0], ylim[i][1])
        axs_axial[i, 0].set_zorder(1)
        axs_axial[i, 0].patch.set_visible(False)
        axs_axial_twin[i].yaxis.set_tick_params(direction='in', width=2, pad=3)

    # style axes, add x and y labels
    for i in range(3):
        style_axes(axs_axial[i, 0], 'Humerothoracic Elevation (Deg)' if i == 2 else None, 'Axial Rotation (Deg)')
        axs_axial[i, 1].yaxis.set_label_position('right')
        axs_axial[i, 1].yaxis.tick_right()
        style_axes_right(axs_axial[i, 1], 'Humerothoracic Elevation (Deg)' if i == 2 else None, 'SPM{t}')
        axs_axial[i, 1].yaxis.set_tick_params(direction='in', width=2, pad=3)

    fig_norm = plt.figure(figsize=(120 / 25.4, 190 / 25.4))
    ax_norm = fig_norm.subplots(3, 1)
    for ax in ax_norm:
        ax.axhline(0.05, ls='--', color='grey')
        style_axes(ax, 'Humerothoracic Elevation (Deg)', 'p-value')

    # plot
    leg_patch_mean = []
    leg_patch_t = []
    alpha_patch = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        all_traj_true_ht = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_true_gh = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'true_axial_rot', None, 'up']), axis=0)
        all_traj_true_st = all_traj_true_ht - all_traj_true_gh
        all_traj_gh_elev = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'euler.gh_isb', 1, 'up']), axis=0)
        all_traj_st_elev = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'euler.st_isb', 1, 'up']), axis=0)

        # means and standard deviations
        true_mean_ht = np.rad2deg(np.mean(all_traj_true_ht, axis=0))
        true_mean_gh = np.rad2deg(np.mean(all_traj_true_gh, axis=0))
        true_mean_st = np.rad2deg(np.mean(all_traj_true_st, axis=0))
        true_sd_ht = np.rad2deg(np.std(all_traj_true_ht, ddof=1, axis=0))
        true_sd_gh = np.rad2deg(np.std(all_traj_true_gh, ddof=1, axis=0))
        true_sd_st = np.rad2deg(np.std(all_traj_true_st, ddof=1, axis=0))

        gh_elev_mean = np.rad2deg(np.mean(all_traj_gh_elev, axis=0))
        st_elev_mean = np.rad2deg(np.mean(all_traj_st_elev, axis=0))

        # spm
        ht_zero = spm_test(all_traj_true_ht, 0).inference(alpha, two_tailed=True, **infer_params)
        gh_zero = spm_test(all_traj_true_gh, 0).inference(alpha, two_tailed=True, **infer_params)
        st_zero = spm_test(all_traj_true_st, 0).inference(alpha, two_tailed=True, **infer_params)

        print('Activity: {}'.format(activity))
        print('HT')
        print(extract_sig(ht_zero, x))
        print('Min axial rotation: {:.2f} max axial rotation: {:.2f}'.
              format(np.min(true_mean_ht), np.max(true_mean_ht)))
        print('GH')
        print(extract_sig(gh_zero, x))
        print('Min axial rotation: {:.2f} max axial rotation: {:.2f}'.
              format(np.min(true_mean_gh), np.max(true_mean_gh)))
        print('ST')
        print(extract_sig(st_zero, x))
        print('Min axial rotation: {:.2f} max axial rotation: {:.2f}'.
              format(np.min(true_mean_st), np.max(true_mean_st)))

        cur_row = act_row[activity.lower()]
        # plot SHR
        axial_shr = true_mean_gh / true_mean_st
        axial_shr_line = axs_axial_twin[cur_row].plot(x, axial_shr, color=color_map.colors[6])
        print('AXIAL SHR')
        print('Min axial SHR: {:.2f} max axial SHR: {:.2f}'.
              format(np.min(axial_shr), np.max(axial_shr)))

        print('AXIAL SHR THRESHOLD')
        if activity == 'FE':
            print(x[np.nonzero(axial_shr > -1)[0][0]])
        else:
            print(x[np.nonzero(axial_shr < 1)[0][0]])

        elev_shr = gh_elev_mean / st_elev_mean
        elev_shr_line = axs_axial_twin[cur_row].plot(x, elev_shr, color=color_map.colors[7])

        print('ELEV SHR')
        print('Min elevation SHR: {:.2f} max elevation SHR: {:.2f}'.
              format(np.min(elev_shr), np.max(elev_shr)))

        # plot mean +- sd
        true_gh_ln = mean_sd_plot(axs_axial[cur_row, 0], x, true_mean_gh, true_sd_gh,
                                  dict(color=color_map.colors[0], alpha=0.2, hatch='...'),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        true_st_ln = mean_sd_plot(axs_axial[cur_row, 0], x, true_mean_st, true_sd_st,
                                  dict(color=color_map.colors[1], alpha=0.27),
                                  dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        true_ht_ln = mean_sd_plot(axs_axial[cur_row, 0], x, true_mean_ht, true_sd_ht,
                                  dict(color=color_map.colors[2], alpha=0.27),
                                  dict(color=color_map.colors[2], marker=markers[2], markevery=20))

        # plot spm
        ht_t_ln, ht_alpha = spm_plot_alpha(axs_axial[cur_row, 1], x, ht_zero,
                                           dict(color=color_map.colors[2], alpha=0.25), dict(color=color_map.colors[2]))
        gh_t_ln, gh_alpha = spm_plot_alpha(axs_axial[cur_row, 1], x, gh_zero,
                                           dict(color=color_map.colors[0], alpha=0.25), dict(color=color_map.colors[0]))
        st_t_ln, st_alpha = spm_plot_alpha(axs_axial[cur_row, 1], x, st_zero,
                                           dict(color=color_map.colors[1], alpha=0.25), dict(color=color_map.colors[1]))

        # normality
        ht_norm = spm1d.stats.normality.sw.ttest(all_traj_true_ht)
        gh_norm = spm1d.stats.normality.sw.ttest(all_traj_true_gh)
        st_norm = spm1d.stats.normality.sw.ttest(all_traj_true_st)
        ax_norm[idx].plot(x, ht_norm[1], color=color_map.colors[2])
        ax_norm[idx].plot(x, gh_norm[1], color=color_map.colors[0])
        ax_norm[idx].plot(x, st_norm[1], color=color_map.colors[1])

        if idx == 0:
            leg_patch_mean.append(true_ht_ln[0])
            leg_patch_mean.append(true_gh_ln[0])
            leg_patch_mean.append(true_st_ln[0])
            leg_patch_t.append(ht_t_ln[0])
            leg_patch_t.append(gh_t_ln[0])
            leg_patch_t.append(st_t_ln[0])
            alpha_patch.append((ht_alpha, gh_alpha, st_alpha))

    # figure title and legend
    plt.figure(fig_axial_hum.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.0)
    sup_title = fig_axial_hum.suptitle('HT, GH, and ST True Axial Rotation Comparison', x=0.47, y=0.99,
                                       fontweight='bold')
    sup_title.set_zorder(6)
    plt.subplots_adjust(top=0.93)
    leg_left_up = fig_axial_hum.legend(leg_patch_mean, ['HT', 'GH', 'ST'], loc='upper left', bbox_to_anchor=(0, 1),
                                       ncol=3, handlelength=1.2, handletextpad=0.3, columnspacing=0.4, borderpad=0.2)
    leg_left_up_pos = leg_left_up.get_frame().get_bbox().bounds
    leg_left_down = fig_axial_hum.legend([elev_shr_line[0], axial_shr_line[0]], ['Elev SHR', 'Axial Rot SHR'],
                                         loc='upper left', bbox_to_anchor=(0, 0.97), ncol=2, handlelength=1.2,
                                         handletextpad=0.3, columnspacing=0.4, borderpad=0.2)
    leg_right = fig_axial_hum.legend(
        leg_patch_t + alpha_patch, ['HT = 0', 'GH = 0', 'ST = 0', '$\\alpha=0.05$'], loc='upper right',
        handler_map={tuple: HandlerTupleVertical(ndivide=None)}, bbox_to_anchor=(1, 1), ncol=2, handlelength=1.5,
        handletextpad=0.5, columnspacing=0.75, labelspacing=0.3, borderpad=0.2)

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
    axs_axial[1, 0].arrow(33, -14.5, 0, -12, length_includes_head=True, head_width=2, head_length=2)
    axs_axial[1, 0].text(22, -13, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs_axial[0, 0].get_position().bounds
    fig_axial_hum.text(0.5, y0 + h * 1.04, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[1, 0].get_position().bounds
    fig_axial_hum.text(0.5, y0 + h * 1.04, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[2, 0].get_position().bounds
    fig_axial_hum.text(0.5, y0 + h * 1.04, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    plt.figure(fig_norm.number)
    plt.tight_layout()
    fig_norm.suptitle('Normality tests')
    make_interactive()

    plt.show()
