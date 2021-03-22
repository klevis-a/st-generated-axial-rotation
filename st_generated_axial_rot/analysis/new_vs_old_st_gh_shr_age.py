"""Compare SHR between age group using old SHR (GH Elev/ST UR) and new SHR based on contribution of GH and ST joint to
HT elevation

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
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from functools import partial
    import matplotlib.pyplot as plt
    import numpy as np
    import spm1d
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.plot_utils import style_axes, mean_sd_plot, make_interactive, sig_filter
    from st_generated_axial_rot.analysis.new_vs_old_st_gh_shr import shr_compute
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare SHR between age group using old SHR (GH Elev/ST UR) and new SHR based on '
                                     'contribution of GH and ST joint to HT elevation', __package__, __file__))
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

    # prepare db
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_elev['traj_interp'].apply(add_st_gh_contrib)
    db_elev['old_shr'], db_elev['new_shr'], db_elev['old_shr_interp'], db_elev['new_shr_interp'] = \
        zip(*db_elev.apply(shr_compute, axis=1))

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

    plot_utils.init_graphing(params.backend)
    plt.close('all')

    x = np.arange(0, 100 + 0.25, 0.25)
    fig = plt.figure(figsize=(190 / 25.4, 190 / 25.4), dpi=params.dpi)
    axs = fig.subplots(3, 2)

    for i in range(3):
        for j in range(2):
            style_axes(axs[i, j], 'Humerothoracic Elevation (Deg)' if i == 2 else None,
                       'Elevation (deg)' if j == 0 else None)
            axs[i, j].set_ylim(0, 12)

    leg_mean = []
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        activity_df_lt35 = activity_df.loc[activity_df['age_group'] == '<35']
        activity_df_gt45 = activity_df.loc[activity_df['age_group'] == '>45']

        old_shr_lt35 = np.stack(activity_df_lt35['old_shr_interp'], axis=0)
        new_shr_lt35 = np.stack(activity_df_lt35['new_shr_interp'], axis=0)
        old_shr_gt45 = np.stack(activity_df_gt45['old_shr_interp'], axis=0)
        new_shr_gt45 = np.stack(activity_df_gt45['new_shr_interp'], axis=0)

        # means and standard deviations
        old_shr_mean_lt35 = np.mean(old_shr_lt35, axis=0)
        new_shr_mean_lt35 = np.mean(new_shr_lt35, axis=0)
        old_shr_mean_gt45 = np.mean(old_shr_gt45, axis=0)
        new_shr_mean_gt45 = np.mean(new_shr_gt45, axis=0)

        old_shr_sd_lt35 = np.std(old_shr_lt35, ddof=1, axis=0)
        new_shr_sd_lt35 = np.std(new_shr_lt35, ddof=1, axis=0)
        old_shr_sd_gt45 = np.std(old_shr_gt45, ddof=1, axis=0)
        new_shr_sd_gt45 = np.std(new_shr_gt45, ddof=1, axis=0)

        # spm
        spm_old = spm_test(old_shr_lt35, old_shr_gt45, equal_var=False).inference(alpha, two_tailed=True,
                                                                                  **infer_params)
        spm_new = spm_test(new_shr_lt35, new_shr_gt45, equal_var=False).inference(alpha, two_tailed=True,
                                                                                  **infer_params)

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        old_shr_ln_lt35 = mean_sd_plot(axs[cur_row, 0], x, old_shr_mean_lt35, old_shr_sd_lt35,
                                       dict(color=color_map.colors[0], alpha=0.25),
                                       dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        old_shr_ln_gt45 = mean_sd_plot(axs[cur_row, 0], x, old_shr_mean_gt45, old_shr_sd_gt45,
                                       dict(color=color_map.colors[1], alpha=0.25),
                                       dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        new_shr_ln_lt35 = mean_sd_plot(axs[cur_row, 1], x, new_shr_mean_lt35, new_shr_sd_lt35,
                                       dict(color=color_map.colors[0], alpha=0.25),
                                       dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        new_shr_ln_gt45 = mean_sd_plot(axs[cur_row, 1], x, new_shr_mean_gt45, new_shr_sd_gt45,
                                       dict(color=color_map.colors[1], alpha=0.25),
                                       dict(color=color_map.colors[1], marker=markers[1], markevery=20))

        # plot spm
        x_old_sig = sig_filter(spm_old, x)
        x_new_sig = sig_filter(spm_new, x)
        axs[cur_row, 0].plot(x_old_sig, np.repeat(12, x_old_sig.size), color='k', lw=2)
        axs[cur_row, 1].plot(x_new_sig, np.repeat(12, x_new_sig.size), color='k', lw=2)

        if idx == 0:
            leg_mean.append(old_shr_ln_lt35[0])
            leg_mean.append(old_shr_ln_gt45[0])
            leg_mean.append(new_shr_ln_lt35[0])
            leg_mean.append(new_shr_ln_gt45[0])

    # figure title and legend
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('SHR Comparison', x=0.47, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.93)
    leg_left = fig.legend(leg_mean[:2], ['Euler SHR <35', 'Euler SHR >45'], loc='upper left', bbox_to_anchor=(0, 1),
                          ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, labelspacing=0.3,
                          borderpad=0.2)
    leg_right = fig.legend(leg_mean[2:], ['Contrib SHR <35', 'Contrib SHR >45'], loc='upper right', bbox_to_anchor=(1, 1),
                           ncol=1, handlelength=1.5, handletextpad=0.5, columnspacing=0.75, labelspacing=0.3,
                           borderpad=0.2)

    # add axes titles
    _, y0, _, h = axs[0, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[1, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs[2, 0].get_position().bounds
    fig.text(0.5, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig.savefig(params.fig_file)
    else:
        plt.show()
