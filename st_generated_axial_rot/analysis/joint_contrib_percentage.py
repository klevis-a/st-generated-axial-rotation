"""Compare contributions of the ST and GH joints towards Elevation, Axial Rotation, and PoE for elevation trials
by subject

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
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db, extract_sub_rot_norm
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.plot_utils import make_interactive
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare contributions of the ST and GH joints towards Elevation, Axial Rotation, '
                                     'and PoE for elevation trials by subject', __package__, __file__))
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
        gh_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 0, 'up']), axis=0)
        st_poe = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 0, 'up']), axis=0)

        gh_elev = -np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 1, 'up']), axis=0)
        st_elev = -np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 1, 'up']), axis=0)

        gh_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['gh', 'common_fine_up', 'contribs', 2, 'up']), axis=0)
        st_axial = np.stack(activity_df['traj_interp'].apply(
            extract_sub_rot_norm, args=['st', 'common_fine_up', 'contribs', 2, 'up']), axis=0)

        gh_total = np.abs(gh_poe) + np.abs(gh_elev) + np.abs(gh_axial)
        st_total = np.abs(st_poe) + np.abs(st_elev) + np.abs(st_axial)

        gh_poe_per = (np.abs(gh_poe) / gh_total) * 100
        gh_elev_per = (np.abs(gh_elev) / gh_total) * 100
        gh_axial_per = (np.abs(gh_axial) / gh_total) * 100

        st_poe_per = (np.abs(st_poe) / st_total) * 100
        st_elev_per = (np.abs(st_elev) / st_total) * 100
        st_axial_per = (np.abs(st_axial) / st_total) * 100

        gh_array = [np.abs(gh_poe_per), gh_elev_per, gh_axial_per]
        st_array = [np.abs(st_poe_per), st_elev_per, st_axial_per]

        for axis_idx, contrib_axis in enumerate(contrib_axes):
            fig_gh = plt.figure()
            ax_gh = fig_gh.subplots()
            ax_gh.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            plot_utils.style_axes(ax_gh, 'HT Elevation Angle', 'Percentage (%)')

            fig_st = plt.figure()
            ax_st = fig_st.subplots()
            ax_st.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
            plot_utils.style_axes(ax_st, 'HT Elevation Angle', 'Percentage (%)')

            for trial_idx, (trial_name, age_grp) in enumerate(zip(activity_df['Trial_Name'], activity_df['age_group'])):
                ax_gh.plot(x, gh_array[axis_idx][trial_idx], label='_'.join(trial_name.split('_')[0:2]))
                ax_st.plot(x, st_array[axis_idx][trial_idx], label='_'.join(trial_name.split('_')[0:2]))

            ax_gh.plot(x, np.mean(gh_array[axis_idx], axis=0), color='k', label='Mean')
            ax_st.plot(x, np.mean(st_array[axis_idx], axis=0), color='k', label='Mean')

            fig_gh.tight_layout()
            fig_gh.subplots_adjust(bottom=0.2)
            fig_gh.suptitle(activity + ' ' + contrib_axis + ' GH')
            fig_gh.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                          fontsize=8)
            plt.figure(fig_gh.number)
            make_interactive()

            fig_st.tight_layout()
            fig_st.subplots_adjust(bottom=0.2)
            fig_st.suptitle(activity + ' ' + contrib_axis + ' ST')
            fig_st.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                          fontsize=8)
            plt.figure(fig_st.number)
            make_interactive()

    plt.show()
