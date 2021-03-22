"""Compare different methods of computing ST induced HT axial rotation

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
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    import numpy as np
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.analysis_utils import (st_induced_axial_rot_ang_vel, st_induced_axial_rot_fha,
                                                              st_induced_axial_rot_swing_twist)
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare different methods of computing ST induced HT axial rotation',
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
    db_rot = db.loc[db['Trial_Name'].str.contains('_ERa90_|_ERaR_')].copy()
    prepare_db(db_elev, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev])
    db_rot = ready_er_db(db_rot, params.torso_def, params.scap_lateral, params.erar_endpts, params.era90_endpts,
                         params.dtheta_fine)

    # compute st-induced axial rotation with each of the three different methods
    db_elev['st_induced_ang_vel'] = db_elev[['st', 'ht']].apply(lambda x: st_induced_axial_rot_ang_vel(*x), axis=1)
    db_elev['st_induced_fha'] = db_elev[['st', 'ht']].apply(lambda x: st_induced_axial_rot_fha(*x), axis=1)
    db_elev['st_induced_swing_twist'] = db_elev[['st', 'ht']].apply(lambda x: st_induced_axial_rot_swing_twist(*x),
                                                                    axis=1)
    db_rot['st_induced_ang_vel'] = db_rot[['st', 'ht']].apply(lambda x: st_induced_axial_rot_ang_vel(*x), axis=1)
    db_rot['st_induced_fha'] = db_rot[['st', 'ht']].apply(lambda x: st_induced_axial_rot_fha(*x), axis=1)
    db_rot['st_induced_swing_twist'] = db_rot[['st', 'ht']].apply(lambda x: st_induced_axial_rot_swing_twist(*x),
                                                                  axis=1)

    # compare differences
    db_elev['ang_vel_vs_fha'] = db_elev['st_induced_ang_vel'] - db_elev['st_induced_fha']
    db_elev['ang_vel_vs_swing_twist'] = db_elev['st_induced_ang_vel'] - db_elev['st_induced_swing_twist']
    db_elev['fha_vs_swing_twist'] = db_elev['st_induced_fha'] - db_elev['st_induced_swing_twist']
    db_rot['ang_vel_vs_fha'] = db_rot['st_induced_ang_vel'] - db_rot['st_induced_fha']
    db_rot['ang_vel_vs_swing_twist'] = db_rot['st_induced_ang_vel'] - db_rot['st_induced_swing_twist']
    db_rot['fha_vs_swing_twist'] = db_rot['st_induced_fha'] - db_rot['st_induced_swing_twist']

    db_elev['ang_vel_vs_fha_max'] = \
        db_elev['ang_vel_vs_fha'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_elev['ang_vel_vs_swing_twist_max'] = \
        db_elev['ang_vel_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_elev['fha_vs_swing_twist_max'] = \
        db_elev['fha_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['ang_vel_vs_fha_max'] = \
        db_rot['ang_vel_vs_fha'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['ang_vel_vs_swing_twist_max'] = \
        db_rot['ang_vel_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)
    db_rot['fha_vs_swing_twist_max'] = \
        db_rot['fha_vs_swing_twist'].apply(np.absolute).apply(np.amax).apply(np.rad2deg)

    print('ELEVATION')
    print('Angular Velocity vs FHA Maximum Difference: {:.2f}'.format(db_elev['ang_vel_vs_fha_max'].max()))
    print('Angular Velocity vs Swing Twist Maximum Difference: {:.2f}'.format(db_elev['ang_vel_vs_swing_twist_max']
                                                                              .max()))
    print('FHA vs Swing Twist Maximum Difference: {:.2f}'.format(db_elev['fha_vs_swing_twist_max'].max()))

    print('INTERNAL/EXTERNAL ROTATION')
    print('Angular Velocity vs FHA Maximum Difference: {:.2f}'.format(db_rot['ang_vel_vs_fha_max'].max()))
    print('Angular Velocity vs Swing Twist Maximum Difference: {:.2f}'.format(db_rot['ang_vel_vs_swing_twist_max']
                                                                              .max()))
    print('FHA vs Swing Twist Maximum Difference: {:.2f}'.format(db_rot['fha_vs_swing_twist_max'].max()))
