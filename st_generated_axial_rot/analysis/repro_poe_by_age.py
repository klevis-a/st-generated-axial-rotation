"""Compare retraction/protraction and GH PoE by age for elevation trials.

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
    from functools import partial
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import spm1d
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes,
                                                          sig_filter)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare retraction/protraction and GH PoE by age for elevation trials',
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
        spm_test = partial(spm1d.stats.ttest2, equal_var=False)
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest2
        infer_params = {'force_iterations': True}

    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    act_row = {'ca': 0, 'sa': 1, 'fe': 2}

    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    init_graphing(params.backend)
    plt.close('all')

    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs = fig.subplots(3, 2)

    # set axes limits
    ax_limits = [(20, 70), (-40, 50)]
    for i in range(3):
        for j in range(2):
            axs[i, j].set_ylim(ax_limits[j][0], ax_limits[j][1])
            axs[i, j].yaxis.set_major_locator(ticker.MultipleLocator(10))
            style_axes(axs[i, j], 'Humerothoracic Elevation (Deg)' if i == 2 else None,
                       'ReProtraction (Deg)' if j == 0 else 'PoE (Deg)')

    # plot
    leg_mean = []
    for act_idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        db = db[~db['Trial_Name'].str.contains('|'.join(params.excluded_trials))]
        lt35_df = activity_df[activity_df['age_group'] == '<35']
        gt45_df = activity_df[activity_df['age_group'] == '>45']

        all_traj_repro_lt35 = np.stack(lt35_df['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_fine_up', 'euler.st_isb', 0]), axis=0)
        all_traj_poe_lt35 = np.stack(lt35_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_isb', 0]), axis=0)

        all_traj_repro_gt45 = np.stack(gt45_df['traj_interp'].apply(
            extract_sub_rot, args=['st', 'common_fine_up', 'euler.st_isb', 0]), axis=0)
        all_traj_poe_gt45 = np.stack(gt45_df['traj_interp'].apply(
            extract_sub_rot, args=['gh', 'common_fine_up', 'euler.gh_isb', 0]), axis=0)

        # means and standard deviations
        repro_mean_lt35 = np.rad2deg(np.mean(all_traj_repro_lt35, axis=0))
        poe_mean_lt35 = np.rad2deg(np.mean(all_traj_poe_lt35, axis=0))
        repro_sd_lt35 = np.rad2deg(np.std(all_traj_repro_lt35, ddof=1, axis=0))
        poe_sd_lt35 = np.rad2deg(np.std(all_traj_poe_lt35, ddof=1, axis=0))

        repro_mean_gt45 = np.rad2deg(np.mean(all_traj_repro_gt45, axis=0))
        poe_mean_gt45 = np.rad2deg(np.mean(all_traj_poe_gt45, axis=0))
        repro_sd_gt45 = np.rad2deg(np.std(all_traj_repro_gt45, ddof=1, axis=0))
        poe_sd_gt45 = np.rad2deg(np.std(all_traj_poe_gt45, ddof=1, axis=0))

        # spm
        repro_lt35_vs_gt35 = spm_test(all_traj_repro_lt35, all_traj_repro_gt45).inference(alpha, two_tailed=True,
                                                                                          **infer_params)
        poe_lt35_vs_gt35 = spm_test(all_traj_poe_lt35, all_traj_poe_gt45).inference(alpha, two_tailed=True,
                                                                                    **infer_params)

        # plot mean and SD
        cur_row = act_row[activity.lower()]
        repro_ln_lt35 = mean_sd_plot(axs[cur_row, 0], x, repro_mean_lt35, repro_sd_lt35,
                                     dict(color=color_map.colors[0], alpha=0.25),
                                     dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        poe_ln_lt35 = mean_sd_plot(axs[cur_row, 1], x, poe_mean_lt35, poe_sd_lt35,
                                   dict(color=color_map.colors[0], alpha=0.25),
                                   dict(color=color_map.colors[0], marker=markers[2], markevery=20))

        repro_ln_gt45 = mean_sd_plot(axs[cur_row, 0], x, repro_mean_gt45, repro_sd_gt45,
                                     dict(color=color_map.colors[1], alpha=0.25),
                                     dict(color=color_map.colors[1], marker=markers[0], markevery=20))
        poe_ln_gt45 = mean_sd_plot(axs[cur_row, 1], x, poe_mean_gt45, poe_sd_gt45,
                                   dict(color=color_map.colors[1], alpha=0.25),
                                   dict(color=color_map.colors[1], marker=markers[2], markevery=20))

        # plot SPM
        repro_x_sig = sig_filter(repro_lt35_vs_gt35, x)
        poe_x_sig = sig_filter(poe_lt35_vs_gt35, x)
        axs[cur_row, 0].plot(repro_x_sig, np.repeat(ax_limits[0][1], repro_x_sig.size), color='k', lw=2)
        axs[cur_row, 1].plot(poe_x_sig, np.repeat(ax_limits[1][1], poe_x_sig.size), color='k', lw=2)

        if act_idx == 0:
            leg_mean.append(repro_ln_lt35[0])
            leg_mean.append(repro_ln_gt45[0])
            leg_mean.append(poe_ln_lt35[0])
            leg_mean.append(poe_ln_gt45[0])

    # figure title and legend
    plt.figure(fig.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0)
    fig.suptitle('ST ReProtraction and GH PoE Comparison by Age Group', x=0.5, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend(leg_mean[:2], ['RePro <35', 'RePro >45'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          labelspacing=0.3, borderpad=0.2)
    leg_right = fig.legend(leg_mean[2:], ['PoE <35', 'PoE >45'], loc='upper right', bbox_to_anchor=(1, 1), ncol=1,
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
    for row in axs:
        for ax in row:
            ax.set_xticks(x_ticks)

    # add arrows indicating direction
    # axs[0, 0].arrow(35, -15, 0, -15, length_includes_head=True, head_width=2, head_length=2)
    # axs[0, 0].text(23, -15, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    plt.show()
