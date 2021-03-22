"""Plot HT, ST, and GH contribution to axial rotation and compare ST against GH contributions

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
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import spm1d
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes,
                                                          update_yticks, update_ylabel, output_spm_p, retrieve_bp_stats,
                                                          sig_filter, extract_sig)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import (prepare_db, extract_sub_rot_norm, sub_rot_at_max_elev)
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Plot HT, ST, and GH contribution to axial rotation and compare ST against GH '
                                     'contributions', __package__, __file__))
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
    max_pos = 140

    x_elev = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    x_er = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    gs = fig.add_gridspec(6, 2)
    axs_elev = [fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[2:4, 0]), fig.add_subplot(gs[4:6, 0])]
    axs_er = [fig.add_subplot(gs[:3, 1]), fig.add_subplot(gs[3:6, 1])]

    for i in range(3):
        style_axes(axs_elev[i], 'Humerothoracic Elevation (Deg)' if i == 2 else None, 'Axial Rotation (Deg)')

    for i in range(2):
        style_axes(axs_er[i], 'Motion Completion (%)' if i == 1 else None, 'Axial Rotation (Deg)')

    # plot elevation
    print('ELEVATION')
    leg_elev_mean = []
    spm_y = {'ca': 9, 'sa': 11, 'fe': 30}
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        traj_ht = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['ht', 'common_fine_up', 'contribs', 2, 'up']), axis=0)
        traj_st = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 2, 'up']), axis=0)
        traj_gh = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 2, 'up']), axis=0)

        traj_ht_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['ht', 'contribs', 2, 'up']), axis=0)
        traj_st_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'contribs', 2, 'up']), axis=0)
        traj_gh_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['gh', 'contribs', 2, 'up']), axis=0)

        # means and standard deviations
        ht_mean = np.rad2deg(np.mean(traj_ht, axis=0))
        st_mean = np.rad2deg(np.mean(traj_st, axis=0))
        gh_mean = np.rad2deg(np.mean(traj_gh, axis=0))
        ht_sd = np.rad2deg(np.std(traj_ht, axis=0, ddof=1))
        st_sd = np.rad2deg(np.std(traj_st, axis=0, ddof=1))
        gh_sd = np.rad2deg(np.std(traj_gh, axis=0, ddof=1))

        ht_max_mean = np.rad2deg(np.mean(traj_ht_max, axis=0))
        st_max_mean = np.rad2deg(np.mean(traj_st_max, axis=0))
        gh_max_mean = np.rad2deg(np.mean(traj_gh_max, axis=0))
        ht_max_sd = np.rad2deg(np.std(traj_ht_max, axis=0, ddof=1))
        st_max_sd = np.rad2deg(np.std(traj_st_max, axis=0, ddof=1))
        gh_max_sd = np.rad2deg(np.std(traj_gh_max, axis=0, ddof=1))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        gh_ln = mean_sd_plot(axs_elev[cur_row], x_elev, gh_mean, gh_sd,
                             dict(color=color_map.colors[0], alpha=0.25, hatch='...'),
                             dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_ln = mean_sd_plot(axs_elev[cur_row], x_elev, st_mean, st_sd,
                             dict(color=color_map.colors[1], alpha=0.25),
                             dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        ht_ln = mean_sd_plot(axs_elev[cur_row], x_elev, ht_mean, ht_sd,
                             dict(color=color_map.colors[2], alpha=0.25),
                             dict(color=color_map.colors[2], marker=markers[2], markevery=20))

        axs_elev[cur_row].errorbar(max_pos, gh_max_mean, yerr=gh_max_sd, color=color_map.colors[0],
                                   marker=markers[0], capsize=3)
        axs_elev[cur_row].errorbar(max_pos + 3, st_max_mean, yerr=st_max_sd, color=color_map.colors[1],
                                   marker=markers[1], capsize=3)
        axs_elev[cur_row].errorbar(max_pos - 3, ht_max_mean, yerr=ht_max_sd, color=color_map.colors[2],
                                   marker=markers[2], capsize=3)

        # spm
        if activity.lower() == 'fe':
            st_vs_gh = spm_test(traj_st - traj_gh, 0).inference(alpha, two_tailed=True, **infer_params)
        else:
            st_vs_gh = spm_test(-(traj_st - traj_gh), 0).inference(alpha, two_tailed=True, **infer_params)
        x_sig = sig_filter(st_vs_gh, x_elev)
        axs_elev[cur_row].plot(x_sig, np.repeat(spm_y[activity.lower()], x_sig.size), color='k')

        print('Activity: {}'.format(activity))
        print(extract_sig(st_vs_gh, x_elev))
        print('P-values: ')
        print(output_spm_p(st_vs_gh))
        print('HT Max: {:.2f}'.format(np.abs(ht_max_mean)))
        print('ST Max: {:.2f}'.format(np.abs(st_max_mean)))
        print('GH Max: {:.2f}'.format(np.abs(gh_max_mean)))
        print('Percentage: {:.2f}'.format(st_max_mean / ht_max_mean * 100))

        if idx == 0:
            leg_elev_mean.append(ht_ln[0])
            leg_elev_mean.append(st_ln[0])
            leg_elev_mean.append(gh_ln[0])

    # plot external rotation
    print('\nEXTERNAL ROTATION')
    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        traj_ht = np.stack(activity_df['ht_contribs_interp'], axis=0)[:, :, 2]
        traj_st = np.stack(activity_df['st_contribs_interp'], axis=0)[:, :, 2]
        traj_gh = np.stack(activity_df['gh_contribs_interp'], axis=0)[:, :, 2]

        # means and standard deviations
        ht_mean = np.rad2deg(np.mean(traj_ht, axis=0))
        st_mean = np.rad2deg(np.mean(traj_st, axis=0))
        gh_mean = np.rad2deg(np.mean(traj_gh, axis=0))
        ht_sd = np.rad2deg(np.std(traj_ht, axis=0, ddof=1))
        st_sd = np.rad2deg(np.std(traj_st, axis=0, ddof=1))
        gh_sd = np.rad2deg(np.std(traj_gh, axis=0, ddof=1))

        gh_ln = mean_sd_plot(axs_er[idx_act], x_er, gh_mean, gh_sd,
                             dict(color=color_map.colors[0], alpha=0.25, hatch='...'),
                             dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_ln = mean_sd_plot(axs_er[idx_act], x_er, st_mean, st_sd,
                             dict(color=color_map.colors[1], alpha=0.25),
                             dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        ht_ln = mean_sd_plot(axs_er[idx_act], x_er, ht_mean, ht_sd,
                             dict(color=color_map.colors[2], alpha=0.25),
                             dict(color=color_map.colors[2], marker=markers[2], markevery=20))

        st_percent = (traj_st[:, -1] / traj_ht[:, -1]) * 100
        gh_percent = (traj_gh[:, -1] / traj_ht[:, -1]) * 100
        # inset
        ax_inset = axs_er[idx_act].inset_axes([0.02, 0.12, 0.1, 0.5])
        bp = ax_inset.boxplot(st_percent, widths=0.7)
        ax_inset.yaxis.set_label_position('right')
        ax_inset.yaxis.tick_right()
        ax_inset.xaxis.set_ticks([])
        ax_inset.xaxis.set_ticklabels([])
        ax_inset.patch.set_visible(False)
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_linewidth(2)
        ax_inset.spines['left'].set_visible(False)
        ax_inset.spines['bottom'].set_linewidth(2)
        update_yticks(ax_inset, fontsize=8)
        update_ylabel(ax_inset, '% of HT Axial Rotation', font_size=10)
        arrow_style = '->, head_length=6, head_width=3'
        kw = dict(arrowstyle=arrow_style, color='k', shrinkA=1, shrinkB=1)
        cs = "arc,angleA=-90,angleB=90,armA=45,armB=45,rad=0.0"
        if idx_act == 0:
            arrow_patch = patches.FancyArrowPatch((100, -9.8), (8, -40), connectionstyle=cs, **kw)
        else:
            arrow_patch = patches.FancyArrowPatch((100, -14.6), (8, -35), connectionstyle=cs, **kw)
        axs_er[idx_act].add_patch(arrow_patch)

        st_end_deg = np.rad2deg(np.abs(traj_st[:, -1]))
        gh_end_deg = np.rad2deg(np.abs(traj_gh[:, -1]))
        print(activity)
        print('BP Min: {:.2f}, 25: {:.2f}, Median: {:.2f}, 75: {:.2f}, Max: {:.2f}'.format(*retrieve_bp_stats(bp)))
        print('BP Deg Median: {:.2f}, 25: {:.2f}, 75: {:.2f}, Max: {:.2f}'
              .format(np.median(st_end_deg), *np.quantile(st_end_deg, [0.25, 0.75]), np.max(st_end_deg)))
        print('GH Mean Deg: {:.2f} and Percent: {:.2f}'.format(np.mean(gh_end_deg), np.mean(gh_percent)))
        print('ST Mean Deg: {:.2f} and Percent: {:.2f}'.format(np.mean(st_end_deg), np.mean(st_percent)))

        st_vs_ht = spm_test(traj_st[:, 40:] / traj_ht[:, 40:], 0.10).inference(alpha, two_tailed=True, **infer_params)
        x_sig = sig_filter(st_vs_ht, x_er[40:])
        axs_er[idx_act].plot(x_sig, np.repeat(5, x_sig.size), color='k')
        print(extract_sig(st_vs_ht, x_er[40:]))
        print('P-values: ')
        print(output_spm_p(st_vs_ht))

    # add stats test text
    axs_elev[0].text(25, 10, 'GH > ST', ha='left', fontsize=10, fontweight='bold')
    axs_elev[0].text(107, 10, 'GH < ST', ha='left', fontsize=10, fontweight='bold')
    axs_elev[0].set_ylim(None, 15)

    axs_elev[1].text(25, 12, 'GH > ST', ha='left', fontsize=10, fontweight='bold')
    axs_elev[1].set_ylim(None, 18)

    axs_elev[2].text(70, 31, 'GH < ST', ha='left', fontsize=10, fontweight='bold')
    axs_elev[2].set_ylim(None, 35)

    axs_er[0].text(37, 7, 'ST < 10%', ha='left', fontsize=10, fontweight='bold')
    axs_er[0].set_ylim(None, 15)

    axs_er[1].text(18, 7, 'ST < 10%', ha='left', fontsize=10, fontweight='bold')
    axs_er[1].text(82, 7, 'ST > 10%', ha='left', fontsize=10, fontweight='bold')
    axs_er[1].set_ylim(None, 15)

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('ST and GH Contributions to HT Axial Rotation', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend(leg_elev_mean, ['HT', 'ST', 'GH'], loc='upper left', bbox_to_anchor=(0, 1),
                          ncol=3, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, labelspacing=0.3,
                          borderpad=0.2)

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

    for i in range(3):
        axs_elev[i].set_xticks(x_ticks)
        tick_labels = [str(i) for i in x_ticks]
        tick_labels[-1] = 'Max'
        axs_elev[i].set_xticklabels(tick_labels)

    # add arrows indicating direction
    axs_elev[0].arrow(35, -15, 0, -15, length_includes_head=True, head_width=2, head_length=2)
    axs_elev[0].text(23, -15, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

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
