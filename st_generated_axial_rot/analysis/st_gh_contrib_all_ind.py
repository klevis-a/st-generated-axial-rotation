"""Compare contributions of the ST and GH joint to HT motion on a per-trial basis

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
output_dir: Directory where PDF records should be output.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare contributions of the ST and GH joint to HT motion on a per-trial basis',
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
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    for activity, activity_df in db_elev.groupby('Activity', observed=True):
        # overall
        pdf_file_path = output_path / ('ind_all_' + activity + '_' + params.torso_def +
                                       ('_' + params.scap_lateral) + '.pdf')
        with PdfPages(pdf_file_path) as activity_pdf:
            for trial_name, df_row in activity_df.iterrows():
                st_traj = df_row['traj_interp'].st
                gh_traj = df_row['traj_interp'].gh
                up_down = df_row['up_down_analysis']

                start_idx = up_down.max_run_up_start_idx
                end_idx = up_down.max_run_up_end_idx
                sec = slice(start_idx,  end_idx + 1)

                fig = plt.figure()
                ax = fig.subplots()
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                plot_utils.style_axes(ax, 'HT Elevation Angle', 'Rotation (deg)')
                ht_elev = np.rad2deg(-df_row['ht'].euler.ht_isb[sec, 1])

                st_poe_plt = ax.plot(ht_elev, np.rad2deg(st_traj.contribs[sec, 0]), label='ST PoE')
                st_elev_plt = ax.plot(ht_elev, -np.rad2deg(st_traj.contribs[sec, 1]), label='ST Elev')
                st_axial_plt = ax.plot(ht_elev, np.rad2deg(st_traj.contribs[sec, 2]), label='ST Axial')

                ax.plot(ht_elev, np.rad2deg(gh_traj.contribs[sec, 0]), '--', label='GH PoE',
                        color=st_poe_plt[0]._color)
                ax.plot(ht_elev, -np.rad2deg(gh_traj.contribs[sec, 1]), '--', label='GH Elev',
                        color=st_elev_plt[0]._color)
                ax.plot(ht_elev, np.rad2deg(gh_traj.contribs[sec, 2]), '--', label='GH Axial',
                        color=st_axial_plt[0]._color)

                ax.plot(ht_elev, np.rad2deg(st_traj.contribs[sec, 0] + gh_traj.contribs[sec, 0]), label='PoE Sum',
                        color='k')

                fig.tight_layout()
                fig.subplots_adjust(bottom=0.2)
                fig.suptitle('_'.join(trial_name.split('_')[0:2]))
                fig.legend(ncol=10, handletextpad=0.25, columnspacing=0.5, loc='lower left', fontsize=8)
                activity_pdf.savefig(fig)
                fig.clf()
                plt.close(fig)
