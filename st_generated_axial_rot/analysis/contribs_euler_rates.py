"""Compute PoE, Elevation and true axial rotation from integrating Euler rates

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
from st_generated_axial_rot.common.plot_utils import style_axes, make_interactive

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.analysis_utils_contrib import compute_yxy_from_euler_rates
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compute PoE, Elevation and true axial rotation from integrating Euler rates',
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
    db_elev['ht_contribs_euler_rates'] = db_elev['ht'].apply(compute_yxy_from_euler_rates)
    db_elev['gh_contribs_euler_rates'] = db_elev['gh'].apply(compute_yxy_from_euler_rates)

#%%
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    db_row = db_elev.loc[params.trial_name]
    start_idx = db_row.up_down_analysis.max_run_up_start_idx
    end_idx = db_row.up_down_analysis.max_run_up_end_idx + 1
    sec = slice(start_idx, end_idx)
    ht_elev = -np.rad2deg(db_row['ht'].euler.ht_isb[sec, 1])

    for seg in ['ht', 'gh']:
        euler_int = db_row[seg + '_contribs_euler_rates']
        euler = db_row[seg].euler.yxy_intrinsic

        fig = plt.figure(figsize=(120 / 25.4, 190 / 25.4))
        axs = fig.subplots(3, 1)

        y_labels = ['Plane of Elevation', 'Elevation', 'Axial Rotation']
        for i in range(3):
            style_axes(axs[i], seg.upper() + ' Elevation (Deg)' if i == 2 else None, y_labels[i])

        elev_lns = []
        poe_lns = []
        axial_lns = []

        poe_ln = axs[0].plot(ht_elev, np.rad2deg(euler[sec, 0]))
        elev_ln = axs[1].plot(ht_elev, np.rad2deg(euler[sec, 1]))
        axial_ln = axs[2].plot(ht_elev, np.rad2deg(db_row[seg].true_axial_rot[sec]))

        poe_int_ln = axs[0].plot(ht_elev, np.rad2deg(euler[0, 0] + euler_int[sec, 0]), '--', dashes=(5, 10))
        elev_int_ln = axs[1].plot(ht_elev, np.rad2deg(euler[0, 1] + euler_int[sec, 1]), '--', dashes=(5, 10))
        axia_int_ln = axs[2].plot(ht_elev, np.rad2deg(euler_int[sec, 2]), '--', dashes=(5, 10))

        fig.tight_layout()
        fig.suptitle(seg.upper() + ' Euler Rate Integration', fontweight='bold')
        plt.subplots_adjust(top=0.93)
        fig.legend([poe_ln[0], elev_ln[0], axial_ln[0], poe_int_ln[0], elev_int_ln[0], axia_int_ln[0]],
                   ['PoE', 'Elevation', 'Axial', 'PoE Int', 'Elev Int', 'Axial Int'], ncol=2, handlelength=0.75,
                   handletextpad=0.25, columnspacing=0.5, loc='upper right', fontsize=8)

        make_interactive()
    plt.show()
