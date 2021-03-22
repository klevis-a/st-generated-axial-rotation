"""Check the filling of the O45_004_WFE_t01 trial.

This script checks the filling of the O45_004_WFE_t01 trial. I elected to (slightly) fill this trial because it was
within 1 deg of achieving the minimum range of 25 deg HT elevation, thus allowing us to use this range for all the other
subjects. The filling is based from the linear and angular velocity of the trajectory.

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
"""

if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.analysis.up_down_identify import extract_up_down_min_max
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from logging.config import fileConfig
    import logging

    config_dir = Path(mod_arg_parser('Check O45_004_WFE_t01 filling', __package__, __file__))
    params = get_params(config_dir / 'parameters.json')
    db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
    db = db.loc[['O45_004_WFE_t01']]

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # compute min and max ht elevation for each subject
    prepare_db(db, params.torso_def, params.scap_lateral, params.dtheta_fine, params.dtheta_coarse,
               [params.min_elev, params.max_elev], should_clean=False)
    (db['up_min_ht'], db['up_max_ht'], db['down_min_ht'], db['down_max_ht']) = zip(
        *(db['up_down_analysis'].apply(extract_up_down_min_max)))

    plot_utils.init_graphing(params.backend)
    plot_dirs = [['ht', 'ht_isb', 'HT'], ['gh', 'gh_isb', 'GH'], ['st', 'st_isb', 'ST']]
    for plot_dir in plot_dirs:
        traj = db.loc['O45_004_WFE_t01', plot_dir[0]]
        traj_euler = getattr(traj, 'euler')
        fig = plt.figure(figsize=(14, 7), tight_layout=True)
        ax = fig.subplots(2, 3)
        for i in range(3):
            ax[0, i].plot(np.rad2deg(getattr(traj_euler, plot_dir[1])[:, i]))
            ax[1, i].plot(traj.pos[:, i])
            if i == 0:
                plot_utils.style_axes(ax[0, i], None, 'Orientation (deg)')
                plot_utils.style_axes(ax[1, i], 'Frame Index (Zero-Based)', 'Position (mm)')
            else:
                plot_utils.style_axes(ax[0, i], None, None)
                plot_utils.style_axes(ax[1, i], 'Frame Index (Zero-Based)', None)
        fig.suptitle(plot_dir[2])
        plot_utils.make_interactive()
    plt.show()
