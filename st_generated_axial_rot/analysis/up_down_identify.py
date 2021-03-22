"""Create PDF records of where up/down would be selected using the minimum elevation method and the longest
increasing/decreasing range method for all trials.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
torso_def: Anatomical definition of the torso: v3d for Visual3D definition, isb for ISB definition.
scap_lateral: Landmarks to utilize when defining the scapula's lateral (+Z) axis.
backend: Matplotlib backend to use for plotting (e.g. Qt5Agg, macosx, etc.).
output_dir: Directory where PDF records should be output.
"""

import numpy as np
import matplotlib.pyplot as plt
from st_generated_axial_rot.common import plot_utils
import logging

log = logging.getLogger(__name__)


def extract_up_down_min_max(up_down):
    return (up_down.max_run_up_start_val, up_down.max_run_up_end_val, up_down.max_run_down_start_val,
            up_down.max_run_down_end_val)


def trial_plotter(df_row):

    def plotter():
        fig = plt.figure(tight_layout=True)
        ax = fig.subplots()
        ht_elev = -np.rad2deg(df_row['ht'].euler.ht_isb[:, 1])
        num_frames = ht_elev.size
        ax.plot(ht_elev)
        up_down_analysis = df_row['up_down_analysis']
        ax.plot(up_down_analysis.min_elev_up_idx, up_down_analysis.min_elev_up, 'go', ms=4,
                label='Up: ' + str(up_down_analysis.min_elev_up_idx))
        ax.plot(up_down_analysis.max_elev_idx, up_down_analysis.max_elev, 'ro', ms=4,
                label=str(up_down_analysis.max_elev_idx))
        ax.plot(up_down_analysis.min_elev_down_idx, up_down_analysis.min_elev_down, 'go', ms=4,
                label='Down: ' + str(up_down_analysis.min_elev_down_idx))
        ax.plot(up_down_analysis.max_run_up_start_idx, up_down_analysis.max_run_up_start_val, 'co', ms=4,
                label='Up: ' + str(up_down_analysis.max_run_up_start_idx))
        ax.plot(up_down_analysis.max_run_up_end_idx, up_down_analysis.max_run_up_end_val, 'mo', ms=4,
                label='Up: ' + str(up_down_analysis.max_run_up_end_idx))
        ax.plot(up_down_analysis.max_run_down_start_idx, up_down_analysis.max_run_down_start_val, 'co', ms=4,
                label='Down: ' + str(up_down_analysis.max_run_down_start_idx))
        ax.plot(up_down_analysis.max_run_down_end_idx, up_down_analysis.max_run_down_end_val, 'mo', ms=4,
                label='Down: ' + str(up_down_analysis.max_run_down_end_idx))
        ax.plot(np.nan, 'ko', label='Last Idx:' + str(num_frames - 1))

        ax.legend(loc='upper right')
        plot_utils.update_spines(ax)
        plot_utils.update_xticks(ax, font_size=8)
        plot_utils.update_yticks(ax, fontsize=8)
        plot_utils.update_xlabel(ax, 'Frame Number (Zero-Indexed)', font_size=10)
        plot_utils.update_ylabel(ax, 'Humerothoracic Elevation (deg)', font_size=10)
        fig.suptitle(df_row['Trial_Name'])
        return fig

    return plotter


def save_figures(plotter, pdf_page):
    fig = plotter()
    pdf_page.savefig(fig)
    fig.clf()
    plt.close(fig)


if __name__ == '__main__':
    if __package__ is None:
        print('Use -m option to run this library module as a script.')

    import os
    import distutils.util
    from pathlib import Path
    from matplotlib.backends.backend_pdf import PdfPages
    from st_generated_axial_rot.common.analysis_utils import get_trajs
    from st_generated_axial_rot.common.up_down import analyze_up_down
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Identify up down sections per trial', __package__, __file__))
    params = get_params(config_dir / 'parameters.json')
    if not bool(distutils.util.strtobool(os.getenv('VARS_RETAINED', 'False'))):
        # ready db
        db = create_db(params.biplane_vicon_db_dir, BiplaneViconSubject)
        db = db[~db['Trial_Name'].str.contains('|'.join(params.excluded_trials))]
        db['Trial'].apply(pre_fetch)

    # relevant parameters
    output_path = Path(params.output_dir)

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    if bool(distutils.util.strtobool(params.weighted)):
        db_elev = db.loc[db['Trial_Name'].str.contains('_WCA_|_WSA_|_WFE_')].copy()
    else:
        db_elev = db.loc[db['Trial_Name'].str.contains('_CA_|_SA_|_FE_')].copy()

    # compute min and max ht elevation for each subject
    db_elev['ht'], db_elev['gh'], db_elev['st'] = \
        zip(*db_elev['Trial'].apply(get_trajs, args=[db.attrs['dt'], params.torso_def, params.scap_lateral]))
    db_elev['up_down_analysis'] = db_elev['ht'].apply(analyze_up_down)
    db_elev['plotter'] = db_elev.apply(trial_plotter, axis=1)
    (db_elev['up_min_ht'], db_elev['up_max_ht'], db_elev['down_min_ht'], db_elev['down_max_ht']) = zip(
        *(db_elev['up_down_analysis'].apply(extract_up_down_min_max)))

    plot_utils.init_graphing(params.backend)
    for activity, activity_df in db_elev.groupby('Activity', observed=True):
        pdf_file_path = output_path / (activity + '_endpoints.pdf')
        with PdfPages(pdf_file_path) as activity_pdf:
            activity_df['plotter'].apply(save_figures, args=[activity_pdf])
