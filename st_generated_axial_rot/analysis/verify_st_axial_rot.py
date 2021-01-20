"""Compare how much the scapula contributes to HT axial rotation by computing HT-GH true axial rotation vs
projecting the scapula angular velocity onto the longitudinal axis of the humerus.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
trial_name: Trial name to use for verification.
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import cumtrapz
    from biokinepy.vec_ops import extended_dot
    from st_generated_axial_rot.common.plot_utils import init_graphing
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Verify intuition about ST axial rotation', __package__, __file__))
    params = get_params(config_dir / 'parameters.json')

    # ready db
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject, include_anthro=True)
    db['age_group'] = db['Age'].map(lambda age: '<35' if age < 40 else '>45')
    if params.excluded_trials:
        db = db[~db['Trial_Name'].str.contains('|'.join(params.excluded_trials))]
    db = db.loc[[params.trial_name]]

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    prepare_db(db, params.torso_def, False, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev], should_fill=False, should_clean=False)
    trial = db.loc[params.trial_name]

#%%
    # calculate st true axial rotation "manually"
    st_angvel_proj = extended_dot(trial.st.ang_vel, trial.ht.rot_matrix[:, :, 1])
    st_axial_rot_manual = cumtrapz(st_angvel_proj, dx=trial.st.dt, initial=0)
    st_axial_rot = trial.ht.true_axial_rot - trial.gh.true_axial_rot

    # plot
    start_idx = trial.up_down_analysis.max_run_up_start_idx
    end_idx = trial.up_down_analysis.max_run_up_end_idx
    plt.close('all')
    init_graphing(params.backend)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(np.rad2deg(st_axial_rot[start_idx:end_idx+1]), label='HT-GH')
    ax.plot(np.rad2deg(st_axial_rot_manual[start_idx:end_idx + 1]), label='Manual', ls='--')
    ax.legend(loc='upper left')
    plt.show()
