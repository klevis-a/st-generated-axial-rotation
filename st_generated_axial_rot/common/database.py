from pathlib import Path
import numpy as np
import pandas as pd
from biokinepy.trajectory import PoseTrajectory
from biokinepy.cs import ht_r, change_cs, ht_inv
from lazy import lazy
import quaternion
from typing import Sequence, Union, Type, Callable, Tuple, Any
from .python_utils import NestedDescriptor


BIPLANE_FILE_HEADERS = {'frame': np.int32, 'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                        'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}

TORSO_FILE_HEADERS = {'pos_x': np.float64, 'pos_y': np.float64, 'pos_z': np.float64,
                      'quat_w': np.float64, 'quat_x': np.float64, 'quat_y': np.float64, 'quat_z': np.float64}

LANDMARKS_FILE_HEADERS = {'Landmark': 'string', 'X': np.float64, 'Y': np.float64, 'Z': np.float64}

ACTIVITY_TYPES = ['CA', 'SA', 'FE', 'ERa90', 'ERaR', 'IRaB', 'IRaM', 'Static', 'WCA', 'WSA', 'WFE']
"""Short code that appears in the trial name for the activities that subjects performed."""

MARKERS = ['STRN', 'C7', 'T5', 'T10', 'LSHO', 'LCLAV', 'CLAV', 'RCLAV', 'RSH0', 'RACRM', 'RANGL', 'RSPIN', 'RUPAA',
           'RUPAB', 'RUPAC', 'RUPAD', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RHNDA', 'RHNDB', 'RHNDC', 'RHNDD']
"""Skin markers affixed to subjects, named after the anatomical landmark they represent."""

_subject_anthro_dtype = {'Subject': 'string', 'Dominant_Arm': 'string', 'Gender': 'string', 'Age': np.int32,
                         'Height': np.float64, 'Weight': np.float64, 'BMI': np.float64, 'Armpit_Thickness': np.float64,
                         'Hand_Thickness': np.float64}


def landmark_get_item_method(csv_data: pd.DataFrame, landmark_name: str) -> np.ndarray:
    """Return the landmark data, (3,) numpy array view, associated with landmark_name."""
    return csv_data.loc[landmark_name, 'X':'Z'].to_numpy()


class ViconEndpts:
    """Indices (endpoints) that indicate which of the frames in a Vicon trial correspond to the endpoints of the
    reciprocal biplane fluoroscopy trial.

    Provides lazy (and cached) access to the endpoints. Designed as a mix-in class.

    Attributes
    ----------
    endpts_file: Path
        Path to the CSV file containing the endpoint frame indices.
    """

    def __init__(self, endpts_file: Union[Path, Callable], **kwargs):
        if callable(endpts_file):
            self.endpts_file = endpts_file()
        else:
            self.endpts_file = endpts_file
        assert (self.endpts_file.is_file())
        super().__init__(**kwargs)

    @lazy
    def vicon_endpts(self) -> np.ndarray:
        """Indices (endpoints) that indicate which of the frames in a Vicon trial correspond to the endpoints of the
        reciprocal biplane fluoroscopy trial."""
        _endpts_df = pd.read_csv(self.endpts_file, header=0)
        vicon_endpts = np.squeeze(_endpts_df.to_numpy())
        # the exported values assume that the first vicon frame is 1 but Python uses 0 based indexing
        # the exported values are inclusive but most Python and numpy functions (arange) are exclusive of the stop
        # so that's why the stop value is not modified
        vicon_endpts[0] -= 1
        return vicon_endpts


class TrialDescription:
    """Description (name, activity, and number) of a trial in the Vicon and biplane fluoroscopy database.

    Designed as a mix-in class.

    Attributes
    ----------
    trial_name: str
        The unique trial identifier, e.g. U35_001_CA_t01.
    subject_short
        The shortened subject identifer contained in the `trial_name`, e.g. U35_001.
    activity
        The activity code contained in the `trial_name`, e.g. CA.
    trial_number
        The trial number contained in the `trial_name`, e.g. 1.
    """

    def __init__(self, trial_dir_path: Path, **kwargs):
        # now create trial identifiers
        self.trial_name = trial_dir_path.stem
        self.subject_short, self.activity, self.trial_number = TrialDescription.parse_trial_name(self.trial_name)
        super().__init__(**kwargs)

    @staticmethod
    def parse_trial_name(trial_name: str) -> Tuple[str, str, int]:
        trial_name_split = trial_name.split('_')
        subject = '_'.join(trial_name_split[0:2])
        activity = trial_name_split[2]
        trial_number = int(trial_name_split[3][1:])
        return subject, activity, trial_number


def trial_descriptor_df(subject_name: str, trials: Sequence[TrialDescription]) -> pd.DataFrame:
    """Return a Pandas dataframe that contains commonly used fields from the trials supplied."""
    return pd.DataFrame({'Subject_Name': pd.Series([subject_name] * len(trials), dtype=pd.StringDtype()),
                         'Trial_Name': pd.Series([trial.trial_name for trial in trials], dtype=pd.StringDtype()),
                         'Subject_Short': pd.Series([trial.subject_short for trial in trials], dtype=pd.StringDtype()),
                         'Activity': pd.Categorical([trial.activity for trial in trials], categories=ACTIVITY_TYPES),
                         'Trial_Number': pd.Series([trial.trial_number for trial in trials], dtype=np.int)})


class SubjectDescription:
    """Description (name) of a subject in the Vicon and biplane fluoroscopy database.

    Designed as a mix-in class.

    Attributes
    ----------
    subject_name: pathlib.Path
        The subject identifier.
    """

    def __init__(self, subject_dir_path: Path, **kwargs):
        # subject identifier
        self.subject_name = subject_dir_path.stem
        super().__init__(**kwargs)


class ViconCSTransform:
    """Vicon to biplane fluoroscopy homogeneous coordinate system (CS) transformation for a subject in the Vicon and
    biplane fluoroscopy database.

    All trials for a subject utilize the same CS transformation. Designed as a mix-in class.

    Attributes
    ----------
    f_t_v_file: pathlib.Path
        Path to the file containing the Vicon to biplane fluoroscopy coordinate system transformation data.
    """

    def __init__(self, f_t_v_file: Union[Callable, Path], **kwargs):
        if callable(f_t_v_file):
            self.f_t_v_file = f_t_v_file()
        else:
            self.f_t_v_file = f_t_v_file
        assert(self.f_t_v_file.is_file())
        super().__init__(**kwargs)

    @lazy
    def f_t_v_data(self) -> pd.DataFrame:
        """Pandas dataframe of the CS transformation (expressed as a quaternion and translation) as read from the
        containing file."""
        return pd.read_csv(self.f_t_v_file, header=0)

    @lazy
    def f_t_v(self) -> np.ndarray:
        """Homogeneous CS transformation ((4, 4) numpy array) from the Vicon to the biplane fluoroscopy CS."""
        r = quaternion.as_rotation_matrix(quaternion.from_float_array(self.f_t_v_data.iloc[0, :4].to_numpy()))
        return ht_r(r, self.f_t_v_data.iloc[0, 4:].to_numpy())


class BiplaneViconTrial(TrialDescription, ViconEndpts):
    """A trial that contains both biplane and Vicon data.

    Attributes
    ----------
    humerus_biplane_file: pathlib.Path
        File path to the raw kinematic trajectory for the humerus as derived from biplane fluoroscopy
    scapula_biplane_file: pathlib.Path
        File path to the raw kinematic trajectory for the scapula as derived from biplane fluoroscopy
    humerus_biplane_file_avg_smooth: pathlib.Path
        File path to the smoothed kinematic trajectory for the humerus as derived from biplane fluoroscopy
    scapula_biplane_file_avg_smooth: pathlib.Path
        File path to the smoothed kinematic trajectory for the scapula as derived from biplane fluoroscopy
    torso_vicon_file: pathlib.Path
        File path to the kinematic trajectory for the torso (ISB definition) as derived from skin markers
    torso_vicon_file_v3d: pathlib.Path
        File path to the kinematic trajectory for the torso (V3D definition) as derived from skin markers
    subject: biplane_kine.database.vicon_accuracy.BiplaneViconSubject
        Pointer to the subject that contains this trial.
    """

    def __init__(self, trial_dir: Union[str, Path], subject: 'BiplaneViconSubject', **kwargs):
        self.trial_dir_path = trial_dir if isinstance(trial_dir, Path) else Path(trial_dir)
        self.subject = subject
        super().__init__(trial_dir_path=self.trial_dir_path,
                         endpts_file=lambda: self.trial_dir_path / (self.trial_name + '_vicon_endpts.csv'), **kwargs)
        # file paths
        self.humerus_biplane_file = self.trial_dir_path / (self.trial_name + '_humerus_biplane.csv')
        self.humerus_biplane_file_avg_smooth = self.trial_dir_path / (self.trial_name +
                                                                      '_humerus_biplane_avgSmooth.csv')
        self.scapula_biplane_file = self.trial_dir_path / (self.trial_name + '_scapula_biplane.csv')
        self.scapula_biplane_file_avg_smooth = self.trial_dir_path / (self.trial_name +
                                                                      '_scapula_biplane_avgSmooth.csv')
        self.torso_vicon_file = self.trial_dir_path / (self.trial_name + '_torso.csv')
        self.torso_vicon_file_v3d = self.trial_dir_path / (self.trial_name + '_torso_v3d.csv')

        # make sure the files are actually there
        assert (self.humerus_biplane_file.is_file())
        assert (self.scapula_biplane_file.is_file())
        assert (self.humerus_biplane_file_avg_smooth.is_file())
        assert (self.scapula_biplane_file_avg_smooth.is_file())
        assert (self.torso_vicon_file.is_file())
        assert (self.torso_vicon_file_v3d.is_file())

    @lazy
    def humerus_biplane_data(self) -> pd.DataFrame:
        """Humerus raw biplane data."""
        return pd.read_csv(self.humerus_biplane_file, header=0, dtype=BIPLANE_FILE_HEADERS, index_col='frame')

    @lazy
    def scapula_biplane_data(self) -> pd.DataFrame:
        """Scapula raw biplane data."""
        return pd.read_csv(self.scapula_biplane_file, header=0, dtype=BIPLANE_FILE_HEADERS, index_col='frame')

    @lazy
    def humerus_biplane_data_avg_smooth(self) -> pd.DataFrame:
        """Humerus (average) smoothed biplane data."""
        return pd.read_csv(self.humerus_biplane_file_avg_smooth, header=0,
                           dtype=BIPLANE_FILE_HEADERS, index_col='frame')

    @lazy
    def scapula_biplane_data_avg_smooth(self) -> pd.DataFrame:
        """Scapula (average) smothed biplane data."""
        return pd.read_csv(self.scapula_biplane_file_avg_smooth,
                           header=0, dtype=BIPLANE_FILE_HEADERS, index_col='frame')

    @lazy
    def humerus_quat_fluoro(self) -> np.ndarray:
        """Humerus orientation (as a quaternion) expressed in fluoro reference frame."""
        return self.humerus_biplane_data.iloc[:, 3:].to_numpy()

    @lazy
    def humerus_pos_fluoro(self) -> np.ndarray:
        """Humerus position expressed in fluoro reference frame."""
        return self.humerus_biplane_data.iloc[:, :3].to_numpy()

    @lazy
    def humerus_quat_fluoro_avg_smooth(self) -> np.ndarray:
        """Smoothed humerus orientation (as a quaternion) expressed in fluoro reference frame."""
        return self.humerus_biplane_data_avg_smooth.iloc[:, 3:].to_numpy()

    @lazy
    def humerus_pos_fluoro_avg_smooth(self) -> np.ndarray:
        """Smoothed humerus position expressed in fluoro reference frame."""
        return self.humerus_biplane_data_avg_smooth.iloc[:, :3].to_numpy()

    @lazy
    def humerus_frame_nums(self) -> np.ndarray:
        """Frame numbers for which the humerus was tracked in biplane fluoroscopy."""
        return self.humerus_biplane_data.index.to_numpy()

    @lazy
    def scapula_quat_fluoro(self) -> np.ndarray:
        """Scapula orientation (as a quaternion) expressed in fluoro reference frame."""
        return self.scapula_biplane_data.iloc[:, 3:].to_numpy()

    @lazy
    def scapula_pos_fluoro(self) -> np.ndarray:
        """Scapula position expressed in fluoro reference frame."""
        return self.scapula_biplane_data.iloc[:, :3].to_numpy()

    @lazy
    def scapula_quat_fluoro_avg_smooth(self) -> np.ndarray:
        """Smoothed scapula orientation (as a quaternion) expressed in fluoro reference frame."""
        return self.scapula_biplane_data_avg_smooth.iloc[:, 3:].to_numpy()

    @lazy
    def scapula_pos_fluoro_avg_smooth(self) -> np.ndarray:
        """Smoothed scapula position expressed in fluoro reference frame."""
        return self.scapula_biplane_data_avg_smooth.iloc[:, :3].to_numpy()

    @lazy
    def scapula_frame_nums(self) -> np.ndarray:
        """Frame numbers for which the scapula was tracked in biplane fluoroscopy."""
        return self.scapula_biplane_data.index.to_numpy()

    @lazy
    def torso_vicon_data(self) -> pd.DataFrame:
        """Torso trajectory dataframe."""
        return pd.read_csv(self.torso_vicon_file, header=0, dtype=TORSO_FILE_HEADERS)

    @lazy
    def torso_vicon_data_v3d(self) -> pd.DataFrame:
        """V3D torso trajectory dataframe."""
        return pd.read_csv(self.torso_vicon_file_v3d, header=0, dtype=TORSO_FILE_HEADERS)

    @lazy
    def torso_quat_vicon(self) -> np.ndarray:
        """Torso orientation (as a quaternion) expressed in Vicon reference frame."""
        return self.torso_vicon_data.iloc[:, 3:].to_numpy()

    @lazy
    def torso_pos_vicon(self) -> np.ndarray:
        """Torso position expressed in Vicon reference frame."""
        return self.torso_vicon_data.iloc[:, :3].to_numpy()

    @lazy
    def torso_v3d_quat_vicon(self) -> np.ndarray:
        """V3D torso orientation (as a quaternion) expressed in Vicon reference frame."""
        return self.torso_vicon_data_v3d.iloc[:, 3:].to_numpy()

    @lazy
    def torso_v3d_pos_vicon(self) -> np.ndarray:
        """V3D torso position expressed in Vicon reference frame."""
        return self.torso_vicon_data_v3d.iloc[:, :3].to_numpy()


class BiplaneViconSubject(SubjectDescription, ViconCSTransform):
    """A subject that contains multiple BiplaneVicon trials.

    Attributes
    ----------
    subject_dir_path: pathlib.Path
        Path to directory containing subject data.
    humerus_landmarks_file: pathlib.Path
        File path to the humerus anatomical landmarks (in CT coordinate system).
    scapula_landmarks_file: pathlib.Path
        File path to the scapula anatomical landmarks (in CT coordinate system).
    humerus_stl_smooth_file: pathlib.Path
        File path to the humerus STL.
    scapula_stl_smooth_file: pathlib.Path
        File path to the scapula STL.
    trials: list of biplane_kine.database.biplane_vicon_db.BiplaneViconTrial
        List of trials for the subject.
    """

    def __init__(self, subj_dir: Union[str, Path], trial_class: Type[BiplaneViconTrial] = BiplaneViconTrial, **kwargs):
        self.subject_dir_path = subj_dir if isinstance(subj_dir, Path) else Path(subj_dir)
        def f_t_v_file(): return self.subject_dir_path / 'Static' / (self.subject_name + '_F_T_V.csv')
        super().__init__(subject_dir_path=self.subject_dir_path, f_t_v_file=f_t_v_file, **kwargs)
        # landmarks files
        self.humerus_landmarks_file = self.subject_dir_path / 'Static' / (self.subject_name + '_humerus_landmarks.csv')
        self.scapula_landmarks_file = self.subject_dir_path / 'Static' / (self.subject_name + '_scapula_landmarks.csv')
        self.humerus_stl_smooth_file = self.subject_dir_path / 'Static' / (self.subject_name + '_Humerus_smooth.stl')
        self.scapula_stl_smooth_file = self.subject_dir_path / 'Static' / (self.subject_name + '_Scapula_smooth.stl')
        assert(self.humerus_landmarks_file.is_file())
        assert(self.scapula_landmarks_file.is_file())
        assert(self.humerus_stl_smooth_file.is_file())
        assert(self.scapula_stl_smooth_file.is_file())

        self.trials = [trial_class(folder, self) for
                       folder in self.subject_dir_path.iterdir() if (folder.is_dir() and folder.stem != 'Static')]

    @lazy
    def subject_df(self) -> pd.DataFrame:
        """A Pandas dataframe summarizing the Vicon CSV trials belonging to the subject."""
        df = trial_descriptor_df(self.subject_name, self.trials)
        df['Trial'] = pd.Series(self.trials, dtype=object)
        df['Subject'] = pd.Series([self] * len(self.trials))
        return df

    @lazy
    def humerus_landmarks_data(self) -> pd.DataFrame:
        """Landmarks data for the humerus."""
        return pd.read_csv(self.humerus_landmarks_file, header=0, dtype=LANDMARKS_FILE_HEADERS, index_col='Landmark')

    @lazy
    def scapula_landmarks_data(self) -> pd.DataFrame:
        """Landmarks data for the scapula."""
        return pd.read_csv(self.scapula_landmarks_file, header=0, dtype=LANDMARKS_FILE_HEADERS, index_col='Landmark')

    @lazy
    def humerus_landmarks(self) -> NestedDescriptor:
        """Descriptor that allows landmark indexed ([landmark_name]) access to landmarks data. The indexed access
        returns a (3,) numpy array view."""
        return NestedDescriptor(self.humerus_landmarks_data, landmark_get_item_method)

    @lazy
    def scapula_landmarks(self) -> NestedDescriptor:
        """Descriptor that allows landmark indexed ([landmark_name]) access to landmarks data. The indexed access
        returns a (3,) numpy array view."""
        return NestedDescriptor(self.scapula_landmarks_data, landmark_get_item_method)


def create_db(db_dir: Union[str, Path], subject_class: Any, include_anthro: bool = False, **kwargs) -> pd.DataFrame:
    """Create a Pandas dataframe summarizing the trials contained in the filesystem-based database, passing kwargs to
    the subject constructor."""
    db_path = Path(db_dir)
    subjects = [subject_class(subject_dir, **kwargs) for subject_dir in db_path.iterdir() if subject_dir.is_dir()]
    subject_dfs = [subject.subject_df for subject in subjects]
    db = pd.concat(subject_dfs, ignore_index=True)
    if include_anthro:
        anthro = anthro_db(db_dir)
        db = pd.merge(db, anthro, how='left', left_on='Subject_Name', right_on='Subject')
    db.set_index('Trial_Name', drop=False, inplace=True, verify_integrity=True)
    db.attrs['dt'] = 1/100
    return db


def anthro_db(db_dir: Union[Path, str]) -> pd.DataFrame:
    """Create a Pandas dataframe summarizing subject anthropometrics."""
    db_path = Path(db_dir)
    anthro_file = db_path / 'Subject_Anthropometrics.csv'
    subject_anthro = pd.read_csv(anthro_file, header=0, dtype=_subject_anthro_dtype, index_col='Subject')
    return subject_anthro


def pre_fetch(biplane_vicon_trial: BiplaneViconTrial) -> None:
    """Retrieve all data for the trial from disk."""
    # pre-fetch humerus
    biplane_vicon_trial.humerus_quat_fluoro
    biplane_vicon_trial.humerus_pos_fluoro
    biplane_vicon_trial.humerus_quat_fluoro_avg_smooth
    biplane_vicon_trial.humerus_pos_fluoro_avg_smooth
    biplane_vicon_trial.humerus_frame_nums

    # pre-fetch scapula
    biplane_vicon_trial.scapula_quat_fluoro
    biplane_vicon_trial.scapula_pos_fluoro
    biplane_vicon_trial.scapula_quat_fluoro_avg_smooth
    biplane_vicon_trial.scapula_pos_fluoro_avg_smooth
    biplane_vicon_trial.scapula_frame_nums

    # torso
    biplane_vicon_trial.torso_quat_vicon
    biplane_vicon_trial.torso_pos_vicon
    biplane_vicon_trial.torso_v3d_quat_vicon
    biplane_vicon_trial.torso_v3d_pos_vicon


def trajectories_from_trial(trial: BiplaneViconTrial, dt: float, smoothed: bool = True, base_cs: str = 'vicon',
                            torso_def: str = 'isb', frame_sync: bool = True) -> Tuple[PoseTrajectory, PoseTrajectory,
                                                                                      PoseTrajectory]:
    """Create torso, scapula, and humerus trajectories from a trial."""
    assert(np.array_equal(trial.humerus_frame_nums, trial.scapula_frame_nums))

    scap_quat_field = 'scapula_quat_fluoro_avg_smooth' if smoothed else 'scapula_quat_fluoro'
    scap_pos_field = 'scapula_pos_fluoro_avg_smooth' if smoothed else 'scapula_pos_fluoro'
    hum_quat_field = 'humerus_quat_fluoro_avg_smooth' if smoothed else 'humerus_quat_fluoro'
    hum_pos_field = 'humerus_pos_fluoro_avg_smooth' if smoothed else 'humerus_pos_fluoro'

    def get_torso_pos_quat(t, thorax_def):
        if thorax_def == 'isb':
            return t.torso_pos_vicon, t.torso_quat_vicon
        elif thorax_def == 'v3d':
            return t.torso_v3d_pos_vicon, t.torso_v3d_quat_vicon
        else:
            raise ValueError('torso_def must be either isb or v3d.')

    if base_cs == 'vicon':
        # scapula
        scap_rot_mat = quaternion.as_rotation_matrix(quaternion.from_float_array(getattr(trial, scap_quat_field)))
        scap_traj_fluoro = ht_r(scap_rot_mat, getattr(trial, scap_pos_field))
        scap_traj_vicon = change_cs(ht_inv(trial.subject.f_t_v), scap_traj_fluoro)
        scap_traj = PoseTrajectory.from_ht(scap_traj_vicon, dt, trial.scapula_frame_nums)

        # humerus
        hum_rot_mat = quaternion.as_rotation_matrix(quaternion.from_float_array(getattr(trial, hum_quat_field)))
        hum_traj_fluoro = ht_r(hum_rot_mat, getattr(trial, hum_pos_field))
        hum_traj_vicon = change_cs(ht_inv(trial.subject.f_t_v), hum_traj_fluoro)
        hum_traj = PoseTrajectory.from_ht(hum_traj_vicon, dt, trial.humerus_frame_nums)

        # torso
        torso_pos_vicon, torso_quat_vicon = get_torso_pos_quat(trial, torso_def)

        if frame_sync:
            torso_pos_vicon_sync = (torso_pos_vicon[trial.vicon_endpts[0]:
                                                    trial.vicon_endpts[1]])[trial.humerus_frame_nums - 1]
            torso_quat_vicon_sync = (torso_quat_vicon[trial.vicon_endpts[0]:
                                                      trial.vicon_endpts[1]])[trial.humerus_frame_nums - 1]
            torso_traj = PoseTrajectory.from_quat(torso_pos_vicon_sync, torso_quat_vicon_sync, dt,
                                                  trial.humerus_frame_nums)
        else:
            torso_traj = PoseTrajectory.from_quat(torso_pos_vicon, torso_quat_vicon, dt,
                                                  np.arange(torso_pos_vicon.shape[0]) + 1)
    elif base_cs == 'fluoro':
        scap_traj = PoseTrajectory.from_quat(getattr(trial, scap_pos_field), getattr(trial, scap_quat_field), dt,
                                             trial.scapula_frame_nums)
        hum_traj = PoseTrajectory.from_quat(getattr(trial, hum_pos_field), getattr(trial, hum_quat_field), dt,
                                            trial.humerus_frame_nums)

        torso_pos_vicon, torso_quat_vicon = get_torso_pos_quat(trial, torso_def)
        torso_rot_mat = quaternion.as_rotation_matrix(quaternion.from_float_array(torso_quat_vicon))
        torso_traj_vicon = ht_r(torso_rot_mat, torso_pos_vicon)
        torso_traj_fluoro = change_cs(trial.subject.f_t_v, torso_traj_vicon)

        if frame_sync:
            torso_traj_fluoro_sync = (torso_traj_fluoro[trial.vicon_endpts[0]:
                                                        trial.vicon_endpts[1]])[trial.humerus_frame_nums - 1]
            torso_traj = PoseTrajectory.from_ht(torso_traj_fluoro_sync, dt, trial.humerus_frame_nums)
        else:
            torso_traj = PoseTrajectory.from_ht(torso_traj_fluoro, dt, np.arange(torso_traj_fluoro.shape[0]) + 1)
    else:
        raise ValueError('base_cs must be either vicon or fluoro.')

    return torso_traj, scap_traj, hum_traj
