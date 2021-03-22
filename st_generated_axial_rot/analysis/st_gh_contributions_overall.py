"""Compare contributions of the ST and GH joint to HT motion

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
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.ticker as plticker
    from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib
    from st_generated_axial_rot.common import plot_utils
    from st_generated_axial_rot.common.database import create_db, BiplaneViconSubject, pre_fetch
    from st_generated_axial_rot.common.analysis_utils import prepare_db
    from st_generated_axial_rot.common.json_utils import get_params
    from st_generated_axial_rot.common.arg_parser import mod_arg_parser
    from st_generated_axial_rot.common.plot_utils import style_axes
    from st_generated_axial_rot.analysis.overview_elev_trials import (PlotDirective, summary_plotter, ind_plotter,
                                                                      grp_plotter)
    import logging
    from logging.config import fileConfig

    config_dir = Path(mod_arg_parser('Compare contributions of the ST and GH joint to HT motion',
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
    plot_directives = {
        'gh_poe': PlotDirective('gh', 0, 'contribs', 'Plane of elevation (deg)', 'GH PoE Contribution'),
        'st_poe': PlotDirective('st', 0, 'contribs', 'Plane of elevation (deg)', 'ST PoE Contribution'),
        'ht_poe': PlotDirective('ht', 0, 'contribs', 'Plane of elevation (deg)', 'HT PoE Contribution'),
        'gh_elev': PlotDirective('gh', 1, 'contribs', 'Elevation (deg)', 'GH Elevation Contribution'),
        'st_elev': PlotDirective('st', 1, 'contribs', 'Elevation (deg)', 'ST Elevation Contribution'),
        'ht_elev': PlotDirective('ht', 1, 'contribs', 'Elevation (deg)', 'HT Elevation Contribution'),
        'gh_axial': PlotDirective('gh', 2, 'contribs', 'Axial Rotation (deg)', 'GH Axial Rotation Contribution'),
        'st_axial': PlotDirective('st', 2, 'contribs', 'Axial Rotation (deg)', 'ST Axial Rotation Contribution'),
        'ht_axial': PlotDirective('ht', 2, 'contribs', 'Axial Rotation (deg)', 'HT Axial Rotation Contribution')
    }

    age_c = {'>45': 'red', '<35': 'blue'}
    gender_c = {'F': 'red', 'M': 'blue'}
    section = ('ht_ea_up', 'up', 'common_ht_range_coarse', 'common_coarse_up')

    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.tab20c.colors)
    plot_utils.init_graphing(params.backend)
    plt.close('all')
    for activity, activity_df in db_elev.groupby('Activity', observed=True):
        # overall
        pdf_file_path = output_path / ('contrib_' + activity + '_' + params.torso_def +
                                       ('_' + params.scap_lateral) + '.pdf')
        with PdfPages(pdf_file_path) as activity_pdf:
            for name, plot_dir in plot_directives.items():
                fig = plt.figure()
                ax = fig.subplots()
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                _, agg_lines, _ = summary_plotter(
                    activity_df['traj_interp'], plot_dir.traj, section[0], section[1], section[2], section[3],
                    plot_dir.decomp_method, plot_dir.sub_rot, ax, ind_plotter, 'black', 'red')

                ax.legend([agg_lines.lines[0]], [r'Mean$\pm$SE'], loc='lower right', handlelength=0.75,
                          handletextpad=0.25)
                ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10.0))
                style_axes(ax, 'Humerothoracic Elevation (Deg)', plot_dir.y_label)
                fig.tight_layout()
                fig.subplots_adjust(bottom=0.2)
                fig.suptitle(plot_dir.title)
                fig.legend(ncol=10, handlelength=0.75, handletextpad=0.25, columnspacing=0.5, loc='lower left',
                           fontsize=8)
                activity_pdf.savefig(fig)
                fig.clf()
                plt.close(fig)

            # plot by gender
            pdf_file_path = output_path / ('contrib_' + activity + '_' + params.torso_def + ('_' + params.scap_lateral)
                                           + '_up' + '_gender.pdf')
            gender_c = {'F': ['lightcoral', 'red', 'darkorange'], 'M': ['dodgerblue', 'blue', 'navy']}
            grp_plotter(activity_df, 'Gender', pdf_file_path, gender_c, plot_directives, section)

            # plot by age
            pdf_file_path = output_path / ('contrib_' + activity + '_' + params.torso_def + ('_' + params.scap_lateral)
                                           + '_up' + '_age.pdf')
            age_c = {'>45': ['lightcoral', 'red', 'darkorange'], '<35': ['dodgerblue', 'blue', 'navy']}
            grp_plotter(activity_df, 'age_group', pdf_file_path, age_c, plot_directives, section)
