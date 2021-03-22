from typing import NamedTuple, Callable, Union, Tuple
import numpy as np
import pandas as pd
from st_generated_axial_rot.common.analysis_utils import get_trajs, st_induced_axial_rot_fha
from st_generated_axial_rot.common.analysis_utils_contrib import add_st_gh_contrib_ind
from st_generated_axial_rot.common.interpolation import set_long_axis_hum


def interp_axial_rot_er(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int], extract_fnc: Callable,
                        normalize: bool, delta: float = 0.1) -> np.ndarray:
    """Interpolate the axial rotation trajectory specified by traj_name (gh, ht, st), sub_rot (0, 1, 2, None),
    extract_fnc (extraction function), and delta percentage intervals between 0% (Start of Motion) and 100%
    (maximum axial rotation)."""
    y = extract_fnc(df_row, traj_name, sub_rot)
    start_idx = df_row['Start'] - 1
    stop_idx = df_row['Stop'] - 1
    y = y[start_idx:stop_idx+1]
    if normalize:
        y = y - y[0]
    max_idx = df_row['max_idx']
    frame_nums_norm = df_row['frame_nums_norm']
    desired_frames = np.arange(0, 100 + delta, delta)
    return np.interp(desired_frames, frame_nums_norm, y[0:max_idx+1], np.nan, np.nan)


def extract_true(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT or GH true axial rotation."""
    return df_row[traj_name].true_axial_rot


def extract_isb(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT or GH ISB axial rotation."""
    return df_row[traj_name].euler.yxy_intrinsic[:, sub_rot]


def extract_isb_scap(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT or GH ISB axial rotation."""
    return df_row[traj_name].euler.st_isb[:, sub_rot]


def extract_phadke(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT or GH Phadke axial rotation."""
    return df_row[traj_name].euler.xzy_intrinsic[:, sub_rot]


def extract_isb_norm(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract PoE Adjusted ISB axial rotation."""
    return df_row[traj_name].euler.yxy_intrinsic[:, 2] + df_row[traj_name].euler.yxy_intrinsic[:, 0]


def extract_st_induced(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract ST-induced true axial rotation."""
    return df_row[traj_name][:, sub_rot]


def extract_contrib(df_row: pd.Series, traj_name: str, sub_rot: Union[None, int]) -> np.ndarray:
    """Extract HT, ST, or GH contribution."""
    return df_row[traj_name][:, sub_rot]


def compute_interp_range(df_row: pd.Series) -> Tuple[int, np.ndarray]:
    """Compute where maximum external rotation occurs and normalized (0-100) frame numbers."""
    y = df_row.ht.true_axial_rot
    start_idx = df_row['Start'] - 1
    stop_idx = df_row['Stop'] - 1
    y = y[start_idx:stop_idx+1]
    max_idx = np.argmax(np.absolute(y))
    frame_nums = np.arange(0, max_idx+1)
    num_frames = frame_nums[-1] - frame_nums[0] + 1
    # divide by (num_frames - 1) because I want the last frame to be 100
    frame_nums_norm = ((frame_nums - frame_nums[0]) / (num_frames - 1)) * 100
    return max_idx, frame_nums_norm


class PlotDirective(NamedTuple):
    """Class containing directives for what to plot.

    Attributes
    ----------
    traj: str
        Whether to plot HT, ST, or GH.
    extract_fnc: Callable
        Function that extracts the desired metric from traj.
    sub_rot: int, None
        Sub-rotation to extract from trajectory.
    normalize: bool
        Whether to normalize by first point in trajectory.
    y_label: str
        Y label for the plot.
    title: str
        Title string for the plot.
    """
    traj: str
    extract_fnc: Callable
    sub_rot: Union[int, None]
    normalize: bool
    y_label: str
    title: str


plot_directives = {
    'gh_poe_isb': PlotDirective('gh', extract_isb, 0, False, 'Plane of elevation (deg)', 'GH Plane of Elevation ISB'),
    'gh_ea_isb': PlotDirective('gh', extract_isb, 1, False, 'Elevation (deg)', 'GH Elevation ISB'),
    'gh_axial_isb': PlotDirective('gh', extract_isb, 2, True, 'Axial Rotation (deg)', 'GH Axial Rotation ISB'),
    'gh_true_axial': PlotDirective('gh', extract_true, None, True, 'Axial Rotation (deg)', 'GH True Axial Rotation'),
    'ht_poe_isb': PlotDirective('ht', extract_isb, 0, False, 'Plane of elevation (deg)', 'HT Plane of Elevation ISB'),
    'ht_ea_isb': PlotDirective('ht', extract_isb, 1, False, 'Elevation (deg)', 'HT Elevation ISB'),
    'ht_axial_isb': PlotDirective('ht', extract_isb, 2, True, 'Axial Rotation (deg)', 'HT Axial Rotation ISB'),
    'ht_true_axial': PlotDirective('ht', extract_true, None, True, 'Axial Rotation (deg)', 'HT True Axial Rotation'),
    'gh_poe_phadke': PlotDirective('gh', extract_phadke, 1, False, 'Plane of elevation (deg)',
                                   'GH Horz Abd/Flex Phadke'),
    'gh_ea_phadke': PlotDirective('gh', extract_phadke, 0, False, 'Elevation (deg)', 'GH Elevation Phadke'),
    'gh_axial_phadke': PlotDirective('gh', extract_phadke, 2, True, 'Axial Rotation (deg)',
                                     'GH Axial Rotation Phadke'),
    'ht_poe_phadke': PlotDirective('ht', extract_phadke, 1, False, 'Plane of elevation (deg)',
                                   'HT Horz Abd/Flex Phadke'),
    'ht_ea_phadke': PlotDirective('ht', extract_phadke, 0, False, 'Elevation (deg)', 'HT Elevation Phadke'),
    'ht_axial_phadke': PlotDirective('ht', extract_phadke, 2, True, 'Axial Rotation (deg)', 'HT Axial Rotation Phadke'),
    'st_protraction_isb': PlotDirective('st', extract_isb_scap, 0, False, 'Pro/Retraction (deg)', 'ST Pro/Retraction'),
    'st_latmed_isb': PlotDirective('st', extract_isb_scap, 1, False, 'Lateral/Medial (deg)', 'ST Lateral/Medial'),
    'st_tilt_isb': PlotDirective('st', extract_isb_scap, 2, False, 'Tilt (deg)', 'ST Tilt'),
    'ht_axial_isb_norm': PlotDirective('ht', extract_isb_norm, None, True, 'Axial Rotation (deg)',
                                       "Normalized ISB HT (yx'y'') Axial Rotation"),
    'gh_axial_isb_norm': PlotDirective('gh', extract_isb_norm, None, True, 'Axial Rotation (deg)',
                                       "Normalized ISB GH (yx'y'') Axial Rotation"),
    'st_induced_latmed': PlotDirective('st_induced', extract_st_induced, 0, True, 'Axial Rotation (deg)',
                                       'LatMed ST Induced Axial Rotation'),
    'st_induced_repro': PlotDirective('st_induced', extract_st_induced, 1, True, 'Axial Rotation (deg)',
                                      'RePro ST Induced Axial Rotation'),
    'st_induced_tilt': PlotDirective('st_induced', extract_st_induced, 2, True, 'Axial Rotation (deg)',
                                     'Tilt ST Induced Axial Rotation'),
    'st_induced_total': PlotDirective('st_induced', extract_st_induced, 3, True, 'Axial Rotation (deg)',
                                      'Total ST Induced Axial Rotation')}


def ready_er_db(db: pd.DataFrame, torso_def: str, scap_lateral: str, erar_endpts_file: str, era90_endpts_file: str,
                dtheta_fine: float) -> pd.DataFrame:
    """Ready external rotation database for analysis."""
    db_er = db.loc[db['Trial_Name'].str.contains('_ERa90_|_ERaR_')].copy()
    db_er['ht'], db_er['gh'], db_er['st'] = \
        zip(*db_er['Trial'].apply(get_trajs, args=[db_er.attrs['dt'], torso_def, scap_lateral]))
    db_er['ht'].apply(set_long_axis_hum)
    db_er['gh'].apply(set_long_axis_hum)
    db_er['st_induced'] = db_er[['st', 'ht']].apply(lambda x: st_induced_axial_rot_fha(*x), axis=1)

    # add endpoints
    db_erar = db_er.loc[db_er['Trial_Name'].str.contains('_ERaR_')].copy()
    db_era90 = db_er.loc[db_er['Trial_Name'].str.contains('_ERa90_')].copy()
    erar_endpts = pd.read_csv(erar_endpts_file, index_col='Subject')
    era90_endpts = pd.read_csv(era90_endpts_file, index_col='Subject')
    db_erar_endpts = pd.merge(db_erar, erar_endpts, how='inner', left_on='Subject_Name', right_on='Subject',
                              left_index=False, right_index=True)
    db_era90_endpts = pd.merge(db_era90, era90_endpts, how='inner', left_on='Subject_Name', right_on='Subject',
                               left_index=False, right_index=True)
    db_er_endpts = pd.concat((db_erar_endpts, db_era90_endpts))
    db_er_endpts = db_er_endpts[db_er_endpts['Start'] != -1]
    db_er_endpts['max_idx'], db_er_endpts['frame_nums_norm'] = zip(*db_er_endpts.apply(compute_interp_range, axis=1))

    for dir_name, plot_directive in plot_directives.items():
        db_er_endpts[dir_name] = db_er_endpts.apply(
            interp_axial_rot_er, args=[plot_directive.traj, plot_directive.sub_rot, plot_directive.extract_fnc,
                                       plot_directive.normalize, dtheta_fine], axis=1)

    db_er_endpts['ht_contribs'], db_er_endpts['st_contribs'], db_er_endpts['gh_contribs'] = \
        zip(*db_er_endpts[['ht', 'st', 'gh']].apply(lambda x: add_st_gh_contrib_ind(*x), axis=1))

    def add_contrib_interp(df_row, contrib_def, dtheta_fine):
        temp_contribs = []
        for i in range(4):
            current_contrib = interp_axial_rot_er(df_row, contrib_def, i, extract_contrib, True, dtheta_fine)
            temp_contribs.append(current_contrib)
        return np.stack(temp_contribs, axis=1)

    for dir_name in ['ht_contribs', 'st_contribs', 'gh_contribs']:
        db_er_endpts[dir_name + '_interp'] = db_er_endpts.apply(add_contrib_interp, axis=1,
                                                                args=[dir_name, dtheta_fine])

    return db_er_endpts
