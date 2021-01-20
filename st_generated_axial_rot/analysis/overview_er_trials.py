"""Create an overview of axial rotation for external rotation trials.

The path to a config directory (containing parameters.json) must be passed in as an argument. Within parameters.json the
following keys must be present:

logger_name: Name of the loggger set up in logging.ini that will receive log messages from this script.
biplane_vicon_db_dir: Path to the directory containing the biplane and vicon CSV files.
excluded_trials: Trial names to exclude from analysis.
use_ac: Whether to use the AC or GC landmark when building the scapula CS.
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
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.analysis_er_utils import ready_er_db, plot_directives
    import matplotlib.ticker as plticker
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser("Create an overview of axial rotation for external rotation trials",
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
    use_ac = bool(distutils.util.strtobool(params.use_ac))

    # logging
    fileConfig(config_dir / 'logging.ini', disable_existing_loggers=False)
    log = logging.getLogger(params.logger_name)

    # ready db
    db_er = ready_er_db(db, params.torso_def, use_ac, params.erar_endpts, params.era90_endpts, params.dtheta_fine)

#%%
    x = np.arange(0, 100 + params.dtheta_fine, params.dtheta_fine)
    # we need more colors to encompass all subjects than the default color scale offers
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20c.colors)
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    for activity, activity_df in db_er.groupby('Activity', observed=True):
        # overall
        pdf_file_path = output_path / (activity + '_' + params.torso_def + ('_ac' if use_ac else '_gc') + '.pdf')
        with PdfPages(pdf_file_path) as activity_pdf:
            for dir_name, plot_directive in plot_directives.items():
                fig = plt.figure()
                ax = fig.subplots()
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                plot_utils.style_axes(ax, 'Percent Completion (%)', plot_directive.y_label)
                for trial_name, interp_data in zip(activity_df['Trial_Name'], activity_df[dir_name]):
                    ax.plot(x, np.rad2deg(interp_data), label=trial_name.split('_')[0])
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.2)
                fig.suptitle(activity + ' ' + plot_directive.title)
                fig.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left')
                activity_pdf.savefig(fig)
                fig.clf()
                plt.close(fig)

        # age
        pdf_file_path = output_path / (activity + '_' + params.torso_def + ('_ac' if use_ac else '_gc') + '_age.pdf')
        age_c = {'>45': ['lightcoral', 'red', 'darkorange'], '<35': ['dodgerblue', 'blue', 'navy']}
        age_lines = {'>45': [], '<35': []}
        with PdfPages(pdf_file_path) as activity_pdf:
            for dir_name, plot_directive in plot_directives.items():
                fig = plt.figure()
                ax = fig.subplots()
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                plot_utils.style_axes(ax, 'Percent Completion (%)', plot_directive.y_label)
                for age_group, interp_data in zip(activity_df['age_group'], activity_df[dir_name]):
                    ln = ax.plot(x, np.rad2deg(interp_data), color=age_c[age_group][0])
                    age_lines[age_group].append(ln[0])
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.2)
                fig.suptitle(activity + ' ' + plot_directive.title)
                fig.legend([age_lines['<35'][0], age_lines['>45'][0]], ['<35', '>45'], ncol=2, handlelength=0.75,
                           handletextpad=0.25, columnspacing=0.5, loc='lower left')
                activity_pdf.savefig(fig)
                fig.clf()
                plt.close(fig)

        # gender
        pdf_file_path = output_path / (activity + '_' + params.torso_def + ('_ac' if use_ac else '_gc') + '_gender.pdf')
        gender_c = {'F': ['lightcoral', 'red', 'darkorange'], 'M': ['dodgerblue', 'blue', 'navy']}
        gender_lines = {'F': [], 'M': []}
        with PdfPages(pdf_file_path) as activity_pdf:
            for dir_name, plot_directive in plot_directives.items():
                fig = plt.figure()
                ax = fig.subplots()
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                plot_utils.style_axes(ax, 'Percent Completion (%)', plot_directive.y_label)
                for gender, interp_data in zip(activity_df['Gender'], activity_df[dir_name]):
                    ln = ax.plot(x, np.rad2deg(interp_data), color=gender_c[gender][0])
                    gender_lines[gender].append(ln[0])
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.2)
                fig.suptitle(activity + ' ' + plot_directive.title)
                fig.legend([gender_lines['F'][0], gender_lines['M'][0]], ['Female', 'Male'], ncol=2, handlelength=0.75,
                           handletextpad=0.25, columnspacing=0.5, loc='lower left')
                activity_pdf.savefig(fig)
                fig.clf()
                plt.close(fig)
