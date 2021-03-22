"""Plot each component of ST induced HT axial rotation for each plane of elevation

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
    import matplotlib.ticker as ticker
    import spm1d
    from st_generated_axial_rot.common.plot_utils import (init_graphing, make_interactive, mean_sd_plot, style_axes,
                                                          output_spm_p, sig_filter, extract_sig)
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.analysis_utils import \
        (prepare_db, extract_sub_rot_norm, st_induced_axial_rot_ang_vel, add_st_induced, sub_rot_at_max_elev)
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

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_elev['traj_interp'].apply(add_st_induced, args=[st_induced_axial_rot_ang_vel])
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

    # style axes, add x and y labels
    for i in range(3):
        style_axes(axs_elev[i], 'Humerothoracic Elevation (Deg)' if i == 2 else None, 'Axial Rotation (Deg)')
        axs_elev[i].yaxis.set_major_locator(ticker.MultipleLocator(5))

    for i in range(2):
        style_axes(axs_er[i], 'Motion Completion (%)' if i == 1 else None, 'Axial Rotation (Deg)')
        axs_er[i].yaxis.set_major_locator(ticker.MultipleLocator(5))

    # plot
    print('ELEVATION')
    leg_elev_mean = []
    spm_y = {'ca': 8, 'sa': 6, 'fe': 24}
    for idx, (activity, activity_df) in enumerate(db_elev.groupby('Activity', observed=True)):
        traj_st = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 3, 'up']), axis=0)
        traj_st_x = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 0, 'up']), axis=0)
        traj_st_y = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 1, 'up']), axis=0)
        traj_st_z = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'induced_axial_rot', 2, 'up']), axis=0)

        traj_st_max = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'induced_axial_rot', 3, 'up']), axis=0)
        traj_st_max_x = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'induced_axial_rot', 0, 'up']), axis=0)
        traj_st_max_y = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'induced_axial_rot', 1, 'up']), axis=0)
        traj_st_max_z = np.stack(activity_df['traj_interp'].apply(
            sub_rot_at_max_elev, args=['st', 'induced_axial_rot', 2, 'up']), axis=0)

        # means and standard deviations
        st_mean = np.rad2deg(np.mean(traj_st, axis=0))
        st_mean_x = np.rad2deg(np.mean(traj_st_x, axis=0))
        st_mean_y = np.rad2deg(np.mean(traj_st_y, axis=0))
        st_mean_z = np.rad2deg(np.mean(traj_st_z, axis=0))
        st_sd = np.rad2deg(np.std(traj_st, ddof=1, axis=0))
        st_sd_x = np.rad2deg(np.std(traj_st_x, ddof=1, axis=0))
        st_sd_y = np.rad2deg(np.std(traj_st_y, ddof=1, axis=0))
        st_sd_z = np.rad2deg(np.std(traj_st_z, ddof=1, axis=0))

        st_max_mean = np.rad2deg(np.mean(traj_st_max, axis=0))
        st_max_mean_x = np.rad2deg(np.mean(traj_st_max_x, axis=0))
        st_max_mean_y = np.rad2deg(np.mean(traj_st_max_y, axis=0))
        st_max_mean_z = np.rad2deg(np.mean(traj_st_max_z, axis=0))
        st_max_sd = np.rad2deg(np.std(traj_st_max, ddof=1, axis=0))
        st_max_sd_x = np.rad2deg(np.std(traj_st_max_x, ddof=1, axis=0))
        st_max_sd_y = np.rad2deg(np.std(traj_st_max_y, ddof=1, axis=0))
        st_max_sd_z = np.rad2deg(np.std(traj_st_max_z, ddof=1, axis=0))

        # plot mean +- sd
        cur_row = act_row[activity.lower()]
        st_ln = mean_sd_plot(axs_elev[cur_row], x_elev, st_mean, st_sd,
                             dict(color=color_map.colors[0], alpha=0.2, hatch='ooo'),
                             dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_x_ln = mean_sd_plot(axs_elev[cur_row], x_elev, st_mean_x, st_sd_x,
                               dict(color=color_map.colors[1], alpha=0.3),
                               dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        st_y_ln = mean_sd_plot(axs_elev[cur_row], x_elev, st_mean_y, st_sd_y,
                               dict(color=color_map.colors[7], alpha=0.3),
                               dict(color=color_map.colors[7], marker=markers[2], markevery=20))
        st_z_ln = mean_sd_plot(axs_elev[cur_row], x_elev, st_mean_z, st_sd_z,
                               dict(color=color_map.colors[3], alpha=0.2, hatch='xxx'),
                               dict(color=color_map.colors[3], marker=markers[3], markevery=20))

        axs_elev[cur_row].errorbar(max_pos, st_max_mean, yerr=st_max_sd, color=color_map.colors[0],
                                   marker=markers[0], capsize=3)
        axs_elev[cur_row].errorbar(max_pos - 3, st_max_mean_x, yerr=st_max_sd_x, color=color_map.colors[1],
                                   marker=markers[1], capsize=3)
        axs_elev[cur_row].errorbar(max_pos + 3, st_max_mean_y, yerr=st_max_sd_y, color=color_map.colors[7],
                                   marker=markers[2], capsize=3)
        axs_elev[cur_row].errorbar(max_pos + 6, st_max_mean_z, yerr=st_max_sd_z, color=color_map.colors[3],
                                   marker=markers[3], capsize=3)

        # spm
        if activity.lower() == 'fe':
            latmed_vs_repro = spm_test(traj_st_x - traj_st_y, 0).inference(alpha, two_tailed=True,
                                                                           **infer_params)
            latmed_vs_tilt = spm_test(traj_st_x - traj_st_z, 0).inference(alpha, two_tailed=True,
                                                                          **infer_params)
        else:
            latmed_vs_repro = spm_test(-(traj_st_x - traj_st_y), 0).inference(alpha, two_tailed=True,
                                                                              **infer_params)
            latmed_vs_tilt = spm_test(-(traj_st_x - traj_st_z), 0).inference(alpha, two_tailed=True,
                                                                             **infer_params)

        x_sig_repro = sig_filter(latmed_vs_repro, x_elev)
        x_sig_tilt = sig_filter(latmed_vs_tilt, x_elev)
        axs_elev[cur_row].plot(x_sig_repro, np.repeat(spm_y[activity.lower()], x_sig_repro.size), '-', color='k')
        axs_elev[cur_row].plot(x_sig_tilt, np.repeat(spm_y[activity.lower()] - 3, x_sig_tilt.size), '-', color='k')

        print('Activity: {}'.format(activity))
        print('Repro Sig')
        print(extract_sig(latmed_vs_repro, x_elev))
        print('Tilt Sig')
        print(extract_sig(latmed_vs_tilt, x_elev))
        print('P-values: ')
        print(output_spm_p(latmed_vs_repro))
        print(output_spm_p(latmed_vs_tilt))
        print('LatMed Max Deg: {:.2f}'.format(st_max_mean_x))
        print('Repro Max Deg: {:.2f}'.format(st_max_mean_y))
        print('Tilt Max Deg: {:.2f}'.format(st_max_mean_z))
        print('Total Max Deg: {:.2f}'.format(st_max_mean))
        print('LatMed Max Percentage: {:.2f}'.format(st_max_mean_x / st_max_mean * 100))
        print('Repro Max Percentage: {:.2f}'.format(st_max_mean_y / st_max_mean * 100))
        print('Tilt Max Percentage: {:.2f}'.format(st_max_mean_z / st_max_mean * 100))

        if idx == 0:
            leg_elev_mean.append(st_ln[0])
            leg_elev_mean.append(st_x_ln[0])
            leg_elev_mean.append(st_y_ln[0])
            leg_elev_mean.append(st_z_ln[0])

    # plot ER
    print('EXTERNAL ROTATION')
    for idx_act, (activity, activity_df) in enumerate(db_er_endpts.groupby('Activity', observed=True)):
        print('Activity: {}'.format(activity))
        traj_latmed = np.stack(activity_df['st_latmed_isb'], axis=0)
        subj_ur = traj_latmed[:, -1] - traj_latmed[:, 0]
        print('UR Mean:{:.2f}'.format(np.rad2deg(np.mean(subj_ur))))
        traj_poe = np.stack(activity_df['gh_poe_isb'], axis=0)
        print('GH PoE Mean:{:.2f}'.format(np.rad2deg(np.mean(traj_poe))))
        traj_st = np.stack(activity_df['st_induced_total'], axis=0)
        traj_st_x = np.stack(activity_df['st_induced_latmed'], axis=0)
        traj_st_y = np.stack(activity_df['st_induced_repro'], axis=0)
        traj_st_z = np.stack(activity_df['st_induced_tilt'], axis=0)

        # means and standard deviations
        st_mean = np.rad2deg(np.mean(traj_st, axis=0))
        st_mean_x = np.rad2deg(np.mean(traj_st_x, axis=0))
        st_mean_y = np.rad2deg(np.mean(traj_st_y, axis=0))
        st_mean_z = np.rad2deg(np.mean(traj_st_z, axis=0))
        st_sd = np.rad2deg(np.std(traj_st, ddof=1, axis=0))
        st_sd_x = np.rad2deg(np.std(traj_st_x, ddof=1, axis=0))
        st_sd_y = np.rad2deg(np.std(traj_st_y, ddof=1, axis=0))
        st_sd_z = np.rad2deg(np.std(traj_st_z, ddof=1, axis=0))

        # plot mean +- sd
        st_ln = mean_sd_plot(axs_er[idx_act], x_er, st_mean, st_sd,
                             dict(color=color_map.colors[0], alpha=0.2, hatch='ooo'),
                             dict(color=color_map.colors[0], marker=markers[0], markevery=20))
        st_x_ln = mean_sd_plot(axs_er[idx_act], x_er, st_mean_x, st_sd_x,
                               dict(color=color_map.colors[1], alpha=0.3),
                               dict(color=color_map.colors[1], marker=markers[1], markevery=20))
        st_y_ln = mean_sd_plot(axs_er[idx_act], x_er, st_mean_y, st_sd_y,
                               dict(color=color_map.colors[7], alpha=0.3),
                               dict(color=color_map.colors[7], marker=markers[2], markevery=20))
        st_z_ln = mean_sd_plot(axs_er[idx_act], x_er, st_mean_z, st_sd_z,
                               dict(color=color_map.colors[3], alpha=0.2, hatch='xxx'),
                               dict(color=color_map.colors[3], marker=markers[3], markevery=20))

        # spm
        if activity.lower() == 'era90':
            latmed_vs_repro = \
                spm_test(-(traj_st_x[:, 1:] - traj_st_y[:, 1:]), 0).inference(alpha, two_tailed=True, **infer_params)
            latmed_vs_tilt = \
                spm_test(-(traj_st_x[:, 1:] - traj_st_z[:, 1:]), 0).inference(alpha, two_tailed=True, **infer_params)
            x_sig_repro = sig_filter(latmed_vs_repro, x_er[1:])
            x_sig_tilt = sig_filter(latmed_vs_tilt, x_er[1:])

            axs_er[idx_act].plot(x_sig_repro, np.repeat(3.5, x_sig_repro.size), color='k')
            axs_er[idx_act].plot(x_sig_tilt, np.repeat(2, x_sig_tilt.size), color='k')

            print('Repro Sig')
            print(extract_sig(latmed_vs_repro, x_er[1:]))
            print('Tilt Sig')
            print(extract_sig(latmed_vs_tilt, x_er[1:]))
            print('P-values: ')
            print(output_spm_p(latmed_vs_repro))
            print(output_spm_p(latmed_vs_tilt))
        else:
            repro_vs_latmed = spm_test(-(traj_st_y[:, 1:] - traj_st_x[:, 1:]), 0).inference(alpha, two_tailed=True,
                                                                                            **infer_params)
            repro_vs_tilt = spm_test(-(traj_st_y[:, 1:] - traj_st_z[:, 1:]), 0).inference(alpha, two_tailed=True,
                                                                                          **infer_params)
            x_sig_latmed = sig_filter(repro_vs_latmed, x_er[1:])
            x_sig_tilt = sig_filter(repro_vs_tilt, x_er[1:])

            axs_er[idx_act].plot(x_sig_latmed, np.repeat(3.25, x_sig_latmed.size), color='k')
            axs_er[idx_act].plot(x_sig_tilt, np.repeat(2, x_sig_tilt.size), color='k')

            print('LatMed Sig')
            print(extract_sig(repro_vs_latmed, x_er[1:]))
            print('Tilt Sig')
            print(extract_sig(repro_vs_tilt, x_er[1:]))
            print('P-values: ')
            print(output_spm_p(repro_vs_latmed))
            print(output_spm_p(repro_vs_tilt))

        print('LatMed Max Deg: {:.2f}'.format(st_mean_x[-1]))
        print('Repro Max Deg: {:.2f}'.format(st_mean_y[-1]))
        print('Tilt Max Deg: {:.2f}'.format(st_mean_z[-1]))
        print('Total Max Deg: {:.2f}'.format(st_mean[-1]))
        print('LatMed Max Percentage: {:.2f}'.format(st_mean_x[-1] / st_mean[-1] * 100))
        print('Repro Max Percentage: {:.2f}'.format(st_mean_y[-1] / st_mean[-1] * 100))
        print('Tilt Max Percentage: {:.2f}'.format(st_mean_z[-1] / st_mean[-1] * 100))

    # add stats test text
    axs_elev[0].text(102, 8.5, 'Upward Rot > RePro', ha='center', fontsize=10, fontweight='bold')
    axs_elev[0].text(104, 5.5, 'Upward Rot > Tilt', ha='center', fontsize=10, fontweight='bold')
    axs_elev[0].set_ylim(None, 11.5)

    axs_elev[1].text(110, 6.5, 'Upward Rot > RePro', ha='center', fontsize=10, fontweight='bold')
    axs_elev[1].text(45, 4, 'Upward Rot < Tilt', ha='center', fontsize=10, fontweight='bold')
    axs_elev[1].set_ylim(None, 9)

    axs_elev[2].text(90, 24.5, 'Upward Rot > RePro', ha='center', fontsize=10, fontweight='bold')
    axs_elev[2].text(80, 21.5, 'Upward Rot > Tilt', ha='center', fontsize=10, fontweight='bold')
    axs_elev[2].set_ylim(None, 28)

    axs_er[0].text(75, 3.5, 'RePro > Upward Rot', ha='center', fontsize=10, fontweight='bold')
    axs_er[0].text(75, 2.25, 'RePro > Tilt', ha='center', fontsize=10, fontweight='bold')
    axs_er[0].set_ylim(None, 5)

    axs_er[1].text(55, 3.75, 'Upward Rot > RePro', ha='center', fontsize=10, fontweight='bold')
    axs_er[1].text(70, 2.25, 'Upward Rot > Tilt', ha='center', fontsize=10, fontweight='bold')
    axs_er[1].set_ylim(None, 5)

    # figure title and legend
    plt.figure(fig.number)
    plt.tight_layout(pad=0.25, h_pad=1.5, w_pad=0.5)
    fig.suptitle('Components of ST-generated Axial Rotation', x=0.53, y=0.99, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    leg_left = fig.legend(leg_elev_mean, ['Total', 'Upward Rot', 'RePro', 'Tilt'], loc='upper left',
                          bbox_to_anchor=(0, 1), ncol=2, handlelength=1.5, handletextpad=0.5,
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
    x_ticks = np.concatenate((x_ticks, [max_pos]))

    for i in range(3):
        axs_elev[i].set_xticks(x_ticks)
        tick_labels = [str(i) for i in x_ticks]
        tick_labels[-1] = 'Max'
        axs_elev[i].set_xticklabels(tick_labels)

    # add arrows indicating direction
    axs_elev[0].arrow(35, -10, 0, -10, length_includes_head=True, head_width=2, head_length=2)
    axs_elev[0].text(23, -10, 'External\nRotation', rotation=90, va='top', ha='left', fontsize=10)

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
