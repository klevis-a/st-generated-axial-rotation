"""Simple convergence plot on a per-trial basis

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

trial_name: The trial to plot.
logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
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

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import cumtrapz
    from biokinepy.vec_ops import extended_dot
    from st_generated_axial_rot.common.plot_utils import init_graphing, make_interactive
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Simple convergence plot on a per-trial basis', __package__, __file__))
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

    prepare_db(db, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev], should_fill=False, should_clean=False)
    trial = db.loc[params.trial_name]

#%%
    end_idx = trial.up_down_analysis.max_run_up_end_idx
    keep_every_vec = np.arange(6) + 1
    indices = np.arange(end_idx+1)
    gh_axial_rots = np.zeros(keep_every_vec.size)
    for keep_every_idx, keep_every in enumerate(keep_every_vec):
        indices_cur = indices[0::keep_every]
        if indices_cur[-1] != indices[-1]:
            indices_cur = np.concatenate((indices[0::keep_every], indices[-1][..., np.newaxis]))

        time = trial.ht.frame_nums[indices_cur] * trial.ht.dt
        gh_angvel_proj = extended_dot(np.squeeze(trial.st.rot_matrix[indices_cur] @
                                                 trial.gh.ang_vel[indices_cur][..., np.newaxis]),
                                      trial.ht.rot_matrix[indices_cur, :, 1])
        gh_axial_rot = cumtrapz(gh_angvel_proj, time, initial=0)
        gh_axial_rots[keep_every_idx] = gh_axial_rot[-1]

    # plot
    plt.close('all')
    init_graphing(params.backend)
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(keep_every_vec, np.rad2deg(gh_axial_rots))
    make_interactive()
    plt.show()
