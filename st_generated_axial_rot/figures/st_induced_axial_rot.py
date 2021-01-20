"""Plot each component of ST induced HT axial rotation for each plane of elevation

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
    from st_generated_axial_rot.common.plot_utils import init_graphing, make_interactive, mean_sd_plot, style_axes
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm, st_induced_axial_rot_fha
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Plot each component of ST induced HT axial rotation for each plane of elevation',
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
    db_elev['traj_interp'].apply(st_induced_axial_rot_fha)

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

    fig_axial = plt.figure(figsize=(90 / 25.4, 210 / 25.4), dpi=params.dpi)
    axs_axial = fig_axial.subplots(3, 1)

    # style axes, add x and y labels
    style_axes(axs_axial[0], None, 'Induced Axial Rotation (Deg)')
    style_axes(axs_axial[1], None, 'Induced Axial Rotation (Deg)')
    style_axes(axs_axial[2], 'Humerothoracic Elevation (Deg)', 'Induced Axial Rotation (Deg)')

    # plot
    leg_patch_mean = []
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

        print('Activity: {}'.format(activity))
        print('Total')
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean))))
        print('LatMed')
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean_x))))
        print('RePro')
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean_y))))
        print('Tilt')
        print('Max: {:.2f}'.format(np.max(np.absolute(induced_mean_z))))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        induced_ln = mean_sd_plot(axs_axial[cur_row], x, induced_mean, induced_sd,
                                  dict(color=color_map.colors[0], alpha=0.2, hatch='ooo'),
                                  dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        induced_x_ln = mean_sd_plot(axs_axial[cur_row], x, induced_mean_x, induced_sd_x,
                                    dict(color=color_map.colors[1], alpha=0.3),
                                    dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        induced_y_ln = mean_sd_plot(axs_axial[cur_row], x, induced_mean_y, induced_sd_y,
                                    dict(color=color_map.colors[7], alpha=0.3),
                                    dict(color=color_map.colors[7], marker=markers[2], markevery=20))
        induced_z_ln = mean_sd_plot(axs_axial[cur_row], x, induced_mean_z, induced_sd_z,
                                    dict(color=color_map.colors[3], alpha=0.2, hatch='xxx'),
                                    dict(color=color_map.colors[3], marker=markers[3], markevery=20))

        if idx == 0:
            leg_patch_mean.append(induced_ln[0])
            leg_patch_mean.append(induced_x_ln[0])
            leg_patch_mean.append(induced_y_ln[0])
            leg_patch_mean.append(induced_z_ln[0])

    # figure title and legend
    plt.figure(fig_axial.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig_axial.suptitle('ST Induced HT Axial Rotation', y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    leg_left = fig_axial.legend(leg_patch_mean, ['Total', 'MedLat', 'RePro', 'Tilt'], loc='upper left',
                                bbox_to_anchor=(0, 0.975), ncol=4, handlelength=1.5, handletextpad=0.5,
                                columnspacing=0.75, labelspacing=0.3, borderpad=0.2)

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
    for ax in axs_axial:
        ax.set_xticks(x_ticks)

    # add arrows indicating direction
    axs_axial[0].arrow(35, -10, 0, -10, length_includes_head=True, head_width=2, head_length=2)
    axs_axial[0].text(23, -10, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

    # add axes titles
    _, y0, _, h = axs_axial[0].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Coronal Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[1].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Scapular Plane Abduction', ha='center', fontsize=11, fontweight='bold')

    _, y0, _, h = axs_axial[2].get_position().bounds
    fig_axial.text(0.5, y0 + h * 1.02, 'Forward Elevation', ha='center', fontsize=11, fontweight='bold')

    make_interactive()

    if params.fig_file:
        fig_axial.savefig(params.fig_file)
    else:
        plt.show()
