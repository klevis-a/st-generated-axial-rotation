"""Compare components of ST-induced axial rotation for external rotation trials.

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
    import spm1d
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    import matplotlib.patches as patches
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.plot_utils import (mean_sd_plot, make_interactive, style_axes, spm_plot_alpha,
                                                          HandlerTupleVertical, update_yticks, update_ylabel,
                                                          extract_sig)
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Compare components of ST-induced axial rotation for external rotation trials",
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

    # ready db
    db_er_endpts = ready_er_db(db, params.torso_def, use_ac, params.erar_endpts, params.era90_endpts, params.dtheta_fine)

#%%
    if bool(distutils.util.strtobool(params.parametric)):
        spm_test = spm1d.stats.ttest
        infer_params = {}
    else:
        spm_test = spm1d.stats.nonparam.ttest
        infer_params = {'force_iterations': True}

    x = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    alpha = 0.05
    color_map = plt.get_cmap('Dark2')
    markers = ['^', 'o',  's', '*']
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(2, 2)

    for row_idx, row in enumerate(axs):
        for col_idx, ax in enumerate(row):
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            if col_idx == 0:
                ax.set_ylim(-18 if row_idx == 0 else -23, 5)
                ax.yaxis.set_major_locator(plticker.MultipleLocator(base=5.0))
            x_label = 'Percent Complete (%)' if row_idx == 1 else None
            y_label = 'Axial Rotation (Deg)' if col_idx == 0 else 'SPM{t}'
            style_axes(ax, x_label, y_label)

    mean_lns = []
    t_lns = []
    alpha_lns = []
    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        all_traj_latmed = np.stack(activity_df['st_induced_latmed'], axis=0)
        all_traj_repro = np.stack(activity_df['st_induced_repro'], axis=0)
        all_traj_tilt = np.stack(activity_df['st_induced_tilt'], axis=0)
        all_traj_total = np.stack(activity_df['st_induced_total'], axis=0)

        all_traj_ht = np.stack(activity_df['ht_true'], axis=0)
        max_idx = np.argmax(np.abs(all_traj_ht), axis=1)
        st_percent = (all_traj_total[np.arange(max_idx.size), max_idx] /
                      all_traj_ht[np.arange(max_idx.size), max_idx]) * 100

        # means
        latmed_mean = np.rad2deg(np.mean(all_traj_latmed, axis=0))
        repro_mean = np.rad2deg(np.mean(all_traj_repro, axis=0))
        tilt_mean = np.rad2deg(np.mean(all_traj_tilt, axis=0))
        total_mean = np.rad2deg(np.mean(all_traj_total, axis=0))

        # sds
        latmed_sd = np.rad2deg(np.std(all_traj_latmed, ddof=1, axis=0))
        repro_sd = np.rad2deg(np.std(all_traj_repro, ddof=1, axis=0))
        tilt_sd = np.rad2deg(np.std(all_traj_tilt, ddof=1, axis=0))
        total_sd = np.rad2deg(np.std(all_traj_total, ddof=1, axis=0))

        # plots mean +- sd
        repro_ln = mean_sd_plot(axs[idx_act, 0], x, repro_mean, repro_sd,
                                dict(color=color_map.colors[1], alpha=0.2),
                                dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        total_ln = mean_sd_plot(axs[idx_act, 0], x, total_mean, total_sd,
                                dict(color=color_map.colors[2], alpha=0.2, hatch='ooo'),
                                dict(color=color_map.colors[2], marker=markers[2], markevery=20))
        tilt_ln = mean_sd_plot(axs[idx_act, 0], x, tilt_mean, tilt_sd,
                               dict(color=color_map.colors[7], alpha=0.2),
                               dict(color=color_map.colors[7], marker=markers[3], markevery=20))
        latmed_ln = mean_sd_plot(axs[idx_act, 0], x, latmed_mean, latmed_sd,
                                 dict(color=color_map.colors[0], alpha=0.2, hatch='xxx'),
                                 dict(color=color_map.colors[0], marker=markers[0], markevery=20))

        ax_inset = axs[idx_act, 0].inset_axes([0.02, 0.05, 0.1, 0.5])
        ax_inset.boxplot(st_percent, widths=0.7)
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
        arrow_style = 'Simple, tail_width=0.5, head_width=4, head_length=8'
        kw = dict(arrowstyle=arrow_style, color='k')
        if idx_act == 0:
            arrow_patch = patches.FancyArrowPatch((51.7, -9.8), (20, -12.5), connectionstyle="arc3,rad=-.5", **kw)
        else:
            arrow_patch = patches.FancyArrowPatch((45.3, -14.6), (20, -17), connectionstyle="arc3,rad=-.5", **kw)
        axs[idx_act, 0].add_patch(arrow_patch)

        # spm
        total_vs_zero = spm_test(all_traj_total[:, 1:], 0).inference(alpha, two_tailed=True, **infer_params)
        latmed_vs_zero = spm_test(all_traj_latmed[:, 1:], 0).inference(alpha, two_tailed=True, **infer_params)
        repro_vs_zero = spm_test(all_traj_repro[:, 1:], 0).inference(alpha, two_tailed=True, **infer_params)
        tilt_vs_zero = spm_test(all_traj_tilt[:, 1:], 0).inference(alpha, two_tailed=True, **infer_params)

        # plot spm
        total_t_ln, total_alpha = spm_plot_alpha(axs[idx_act, 1], x[1:], total_vs_zero,
                                                 dict(color=color_map.colors[2], alpha=0.25),
                                                 dict(color=color_map.colors[2]))
        latmed_t_ln, latmed_alpha = spm_plot_alpha(axs[idx_act, 1], x[1:], latmed_vs_zero,
                                                   dict(color=color_map.colors[0], alpha=0.25),
                                                   dict(color=color_map.colors[0]))
        repro_t_ln, repro_alpha = spm_plot_alpha(axs[idx_act, 1], x[1:], repro_vs_zero,
                                                 dict(color=color_map.colors[1], alpha=0.25),
                                                 dict(color=color_map.colors[1]))
        tilt_t_ln, tilt_alpha = spm_plot_alpha(axs[idx_act, 1], x[1:], tilt_vs_zero,
                                               dict(color=color_map.colors[7], alpha=0.25),
                                               dict(color=color_map.colors[7]))
        empty_ln = axs[idx_act, 1].plot(np.nan, color='none')

        # print significance
        print('Activity: {}'.format(activity))
        print('Total')
        print('Max Mean: {:.2f}'.format(np.max(np.abs(total_mean))))
        print(extract_sig(total_vs_zero, x))
        print('LatMed')
        print('Max Mean: {:.2f}'.format(np.max(np.abs(latmed_mean))))
        print(extract_sig(latmed_vs_zero, x))
        print('RePro')
        print('Max Mean: {:.2f}'.format(np.max(np.abs(repro_mean))))
        print(extract_sig(repro_vs_zero, x))
        print('Tilt')
        print('Max Mean: {:.2f}'.format(np.max(np.abs(tilt_mean))))
        print(extract_sig(tilt_vs_zero, x))

        if idx_act == 0:
            # legend lines
            mean_lns.append(repro_ln[0])
            mean_lns.append(total_ln[0])
            mean_lns.append(tilt_ln[0])
            mean_lns.append(latmed_ln[0])

            t_lns.append(repro_t_ln[0])
            t_lns.append(total_t_ln[0])
            t_lns.append(empty_ln[0])
            t_lns.append(tilt_t_ln[0])
            t_lns.append(latmed_t_ln[0])

            alpha_lns.append(repro_alpha)
            alpha_lns.append(total_alpha)
            alpha_lns.append(tilt_alpha)
            alpha_lns.append(latmed_alpha)

    # figure title and legend
    plt.tight_layout(pad=0.5, h_pad=1.5, w_pad=0.5)
    fig.suptitle('ST-induced Axial Rotation for ERaR and ERa90', x=0.48, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.925)
    leg_left = fig.legend(mean_lns, ['RePro', 'Total', 'Tilt', 'LatMed'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=2, handlelength=1.5, handletextpad=0.5, columnspacing=0.75,
                          borderpad=0.2, labelspacing=0.4)
    leg_right = fig.legend(t_lns + [tuple(alpha_lns)],
                           ['RePro=0', 'Total=0', '', 'Tilt=0', 'LatMed=0', '$\\alpha=0.05$'],
                           loc='upper right', handler_map={tuple: HandlerTupleVertical(ndivide=None)},
                           bbox_to_anchor=(1, 1), ncol=2, handlelength=1.2, handletextpad=0.5, columnspacing=0.75,
                           labelspacing=0.3, borderpad=0.2)

    # add arrows indicating direction
    axs[0, 0].arrow(95, -12, 0, -4, length_includes_head=True, head_width=1, head_length=1)
    axs[0, 0].text(85, -12, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERaR ST-induced Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'ERa90 ST-induced Axial Rotation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
