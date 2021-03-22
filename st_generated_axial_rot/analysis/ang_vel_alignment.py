"""Compare ST and GH angular velocity alignment against HT longitudinal axis

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
excluded_trials: Trial names to exclude from analysis.
dtheta_fine: Incremental angle (deg) to use for fine interpolation between minimum and maximum HT elevation analyzed.
dtheta_coarse: Incremental angle (deg) to use for coarse interpolation between minimum and maximum HT elevation analyzed.
min_elev: Minimum HT elevation angle (deg) utilized for analysis that encompasses all trials.
max_elev: Maximum HT elevation angle (deg) utilized for analysis that encompasses all trials.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    from biokinepy.vec_ops import extended_dot
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib, compute_axial_axis
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.plot_utils import make_interactive
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare ST and GH angular velocity alignment against HT longitudinal axis',
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

    # prepare db
    db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_elev['traj_interp'].apply(add_st_gh_contrib)

#%%
    x = db_elev.iloc[0]['traj_interp'].common_ht_range_fine
    contrib_axes = ['PoE', 'Elevation', 'Axial']
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20c.colors)
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    for activity, activity_df in db_elev.groupby('Activity', observed=True):
        fig_gh = plt.figure()
        ax_gh = fig_gh.subplots()
        ax_gh.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
        plot_utils.style_axes(ax_gh, 'HT Elevation Angle', 'Alignment (deg)')

        fig_st = plt.figure()
        ax_st = fig_st.subplots()
        ax_st.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
        plot_utils.style_axes(ax_st, 'HT Elevation Angle', 'Alignment (deg)')

        for trial_name, df_row in activity_df.iterrows():
            age_grp = df_row['age_group']
            ht = df_row['ht']
            st = df_row['st']
            gh = df_row['gh']
            up_down = df_row['up_down_analysis']
            start_idx = up_down.max_run_up_start_idx
            end_idx = up_down.max_run_up_end_idx
            sec = slice(start_idx, end_idx + 1)
            ht_elev = -np.rad2deg(ht.euler.ht_isb[sec, 1])
            elev_axis = compute_axial_axis(ht.rot_matrix)[sec]
            st_ang_vel_axis = st.ang_vel[sec] / np.sqrt(extended_dot(st.ang_vel[sec], st.ang_vel[sec])[..., np.newaxis])
            gh_ang_vel_axis = gh.ang_vel[sec] / np.sqrt(extended_dot(gh.ang_vel[sec], gh.ang_vel[sec])[..., np.newaxis])
            gh_ang_vel_axis = np.squeeze(st.rot_matrix[sec] @ gh_ang_vel_axis[..., np.newaxis])
            ang_st = extended_dot(st_ang_vel_axis, elev_axis)
            ang_gh = extended_dot(gh_ang_vel_axis, elev_axis)
            ax_gh.plot(ht_elev, ang_gh, label='_'.join(trial_name.split('_')[0:2]))
            ax_st.plot(ht_elev, ang_st, label='_'.join(trial_name.split('_')[0:2]))

        fig_gh.tight_layout()
        fig_gh.subplots_adjust(bottom=0.2)
        fig_gh.suptitle(activity + ' GH alignment')
        fig_gh.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                      fontsize=8)
        plt.figure(fig_gh.number)
        make_interactive()

        fig_st.tight_layout()
        fig_st.subplots_adjust(bottom=0.2)
        fig_st.suptitle(activity + ' ST alignment')
        fig_st.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                      fontsize=8)
        plt.figure(fig_st.number)
        make_interactive()

    plt.show()
