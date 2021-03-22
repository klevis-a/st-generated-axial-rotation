"""Compare HT, ST, GH axial rotation for external rotation trials by gender.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
era90_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in 90 deg of abduction trials.
erar_endpts: Path to csv file containing start and stop frames (including both external and internal rotation) for
external rotation in adduction trials.
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
parametric: Whether to use a parametric (true) or non-parametric statistical test (false).
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from functools import partial
    import numpy as np
    import spm1d
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.plot_utils import (mean_sd_plot, make_interactive, style_axes, sig_filter)
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare HT, ST, GH axial rotation for external rotation trials by gender",
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

    # ready db
    db_er_endpts = ready_er_db(db, params.torso_def, params.scap_lateral, params.erar_endpts, params.era90_endpts,
                               params.dtheta_fine)

#%%
    if bool(distutils.util.strtobool(params.parametric)):
        spm_test = partial(spm1d.stats.ttest2, equal_var=False)
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest2
        infer_params = {'force_iterations': True}

    x = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4))
    axs = fig.subplots(2, 3)

    for row_idx, row in enumerate(axs):
        for col_idx, ax in enumerate(row):
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            ax.yaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
            x_label = 'Percent Complete (%)' if row_idx == 1 else None
            y_label = 'Axial Rotation (Deg)' if col_idx == 0 else None
            style_axes(ax, x_label, y_label)

    mean_lns = []
    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        f_df = activity_df[activity_df['Gender'] == 'F']
        m_df = activity_df[activity_df['Gender'] == 'M']

        traj_ht_f = np.stack(f_df['ht_true_axial'], axis=0)
        traj_ht_m = np.stack(m_df['ht_true_axial'], axis=0)

        traj_st_f = np.stack(f_df['st_induced_total'], axis=0)
        traj_st_m = np.stack(m_df['st_induced_total'], axis=0)

        traj_gh_f = np.stack(f_df['gh_true_axial'], axis=0)
        traj_gh_m = np.stack(m_df['gh_true_axial'], axis=0)

        # means
        ht_f_mean = np.rad2deg(np.mean(traj_ht_f, axis=0))
        ht_m_mean = np.rad2deg(np.mean(traj_ht_m, axis=0))
        st_f_mean = np.rad2deg(np.mean(traj_st_f, axis=0))
        st_m_mean = np.rad2deg(np.mean(traj_st_m, axis=0))
        gh_f_mean = np.rad2deg(np.mean(traj_gh_f, axis=0))
        gh_m_mean = np.rad2deg(np.mean(traj_gh_m, axis=0))

        # sds
        ht_f_sd = np.rad2deg(np.std(traj_ht_f, axis=0, ddof=1))
        ht_m_sd = np.rad2deg(np.std(traj_ht_m, axis=0, ddof=1))
        st_f_sd = np.rad2deg(np.std(traj_st_f, axis=0, ddof=1))
        st_m_sd = np.rad2deg(np.std(traj_st_m, axis=0, ddof=1))
        gh_f_sd = np.rad2deg(np.std(traj_gh_f, axis=0, ddof=1))
        gh_m_sd = np.rad2deg(np.std(traj_gh_m, axis=0, ddof=1))

        # plots mean +- sd
        ht_f_ln = mean_sd_plot(axs[idx_act, 0], x, ht_f_mean, ht_f_sd,
                               dict(color=color_map.colors[0], alpha=0.2),
                               dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        ht_m_ln = mean_sd_plot(axs[idx_act, 0], x, ht_m_mean, ht_m_sd,
                               dict(color=color_map.colors[1], alpha=0.2),
                               dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        st_f_ln = mean_sd_plot(axs[idx_act, 1], x, st_f_mean, st_f_sd,
                               dict(color=color_map.colors[0], alpha=0.2),
                               dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_m_ln = mean_sd_plot(axs[idx_act, 1], x, st_m_mean, st_m_sd,
                               dict(color=color_map.colors[1], alpha=0.2),
                               dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        gh_f_ln = mean_sd_plot(axs[idx_act, 2], x, gh_f_mean, gh_f_sd,
                               dict(color=color_map.colors[0], alpha=0.2),
                               dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        gh_m_ln = mean_sd_plot(axs[idx_act, 2], x, gh_m_mean, gh_m_sd,
                               dict(color=color_map.colors[1], alpha=0.2),
                               dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        ht_f_vs_m = spm_test(traj_ht_f[:, 1:], traj_ht_m[:, 1:]).inference(alpha, two_tailed=True, **infer_params)
        st_f_vs_m = spm_test(traj_st_f[:, 1:], traj_st_m[:, 1:]).inference(alpha, two_tailed=True, **infer_params)
        gh_f_vs_m = spm_test(traj_gh_f[:, 1:], traj_gh_m[:, 1:]).inference(alpha, two_tailed=True, **infer_params)
        ht_x_sig = sig_filter(ht_f_vs_m, x[1:])
        st_x_sig = sig_filter(st_f_vs_m, x[1:])
        gh_x_sig = sig_filter(gh_f_vs_m, x[1:])
        axs[idx_act, 0].plot(ht_x_sig, np.repeat(5, ht_x_sig.size), color='k', lw=2)
        axs[idx_act, 1].plot(st_x_sig, np.repeat(5, st_x_sig.size), color='k', lw=2)
        axs[idx_act, 2].plot(gh_x_sig, np.repeat(5, gh_x_sig.size), color='k', lw=2)

        if idx_act == 0:
            # legend lines
            mean_lns.append(ht_f_ln[0])
            mean_lns.append(ht_m_ln[0])
            mean_lns.append(st_f_ln[0])
            mean_lns.append(st_m_ln[0])
            mean_lns.append(gh_f_ln[0])
            mean_lns.append(gh_m_ln[0])

            axs[idx_act, 0].set_title('HT', y=0.9)
            axs[idx_act, 1].set_title('ST', y=0.9)
            axs[idx_act, 2].set_title('GH', y=0.9)

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig.suptitle('ST-induced Axial Rotation for ERaR and ERa90', x=0.48, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.925)
    leg_left = fig.legend(mean_lns[:2], ['F', 'M'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=2, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          borderpad=0.2, labelspacing=0.4)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERaR ST-induced Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERa90 ST-induced Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    plt.show()
