from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import mne
import numpy as np
import scipy.io
from mne.viz import plot_alignment, snapshot_brain_montage
import utils


FeatureName = Literal["raw", "band", "phase", "envelope"]
Band = tuple[float, float]


# =============================================================================
# Paths
# =============================================================================

HUMAN_DATA_DIR = Path("../data/Sedation-RestingState")
MARMOSET_DATA_DIR = Path("../data/riken-auditory-ECoG")


# =============================================================================
# Human EEG metadata
# =============================================================================

@dataclass(frozen=True)
class HumanRecordingInfo:
    file_stem: str
    state_ids: tuple[str, ...]


HUMAN_EEG_RECORDINGS: dict[int, HumanRecordingInfo] = {
    2: HumanRecordingInfo("02-2010-anest 20100210 135.", ("000", "003", "006", "014")),
    3: HumanRecordingInfo("03-2010-anest 20100211 142.", ("003", "008", "021", "026")),
    5: HumanRecordingInfo("05-2010-anest 20100223 095.", ("004", "009", "022", "027")),
    6: HumanRecordingInfo("06-2010-anest 20100224 093.", ("003", "008", "013", "026")),
    7: HumanRecordingInfo("07-2010-anest 20100226 133.", ("003", "008", "021", "027")),
    8: HumanRecordingInfo("08-2010-anest 20100301 095.", ("004", "010", "015", "028")),
    9: HumanRecordingInfo("09-2010-anest 20100301 135.", ("003", "008", "021", "026")),
    10: HumanRecordingInfo("10-2010-anest 20100305 130.", ("005", "010", "015", "028")),
    13: HumanRecordingInfo("13-2010-anest 20100322 132.", ("003", "008", "013", "026")),
    14: HumanRecordingInfo("14-2010-anest 20100324 132.", ("007", "011", "016", "031")),
    18: HumanRecordingInfo("18-2010-anest 20100331 140.", ("003", "009", "014", "027")),
    20: HumanRecordingInfo("20-2010-anest 20100414 131.", ("004", "009", "022", "027")),
    22: HumanRecordingInfo("22-2010-anest 20100415 132.", ("004", "009", "014", "015")),
    23: HumanRecordingInfo("23-2010-anest 20100420 094.", ("003", "008", "022", "027")),
    24: HumanRecordingInfo("24-2010-anest 20100420 134.", ("003", "010", "015", "028")),
    25: HumanRecordingInfo("25-2010-anest 20100422 133.", ("003", "008", "021", "026")),
    26: HumanRecordingInfo("26-2010-anest 20100507 132.", ("003", "008", "013", "026")),
    27: HumanRecordingInfo("27-2010-anest 20100823 104.", ("001", "010", "023", "028")),
    28: HumanRecordingInfo("28-2010-anest 20100824 092.", ("004", "011", "016", "029")),
    29: HumanRecordingInfo("29-2010-anest 20100921 142.", ("005", "010", "023", "028")),
}


HUMAN_EEG_ALL_CHANNELS_91: list[str] = [
    "C3", "C4", "Cz", "E10", "E101", "E102", "E103", "E105", "E106",
    "E109", "E110", "E111", "E112", "E115", "E116", "E117", "E118",
    "E12", "E123", "E13", "E15", "E16", "E18", "E19", "E2", "E20",
    "E23", "E26", "E27", "E28", "E29", "E3", "E30", "E31", "E34",
    "E35", "E37", "E39", "E4", "E40", "E41", "E42", "E46", "E47",
    "E5", "E50", "E51", "E53", "E54", "E55", "E59", "E6", "E60",
    "E61", "E65", "E66", "E67", "E7", "E71", "E72", "E76", "E77",
    "E78", "E79", "E80", "E84", "E85", "E86", "E87", "E90", "E91",
    "E93", "E97", "E98", "F3", "F4", "F7", "F8", "Fp1", "Fp2",
    "Fz", "O1", "O2", "Oz", "P3", "P4", "Pz", "T3", "T4", "T5", "T6",
]


HUMAN_EEG_CHANNEL_SETS: dict[int, list[str]] = {
    5: ["Fp1", "Fp2", "F3", "F4", "C3"],
    19: [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Pz", "Cz",
    ],
    61: [
        "Fp1", "E15", "Fp2", "E26", "E23", "E16", "E3", "E2", "F7",
        "E27", "F3", "E19", "Fz", "E4", "F4", "E123", "F8", "E39",
        "E35", "E29", "E13", "E6", "E112", "E111", "E110", "E115",
        "T3", "E47", "E37", "E31", "Cz", "E80", "E87", "E98", "T4",
        "E50", "P3", "E53", "E54", "E55", "E79", "E86", "P4", "E101",
        "T5", "E59", "E60", "E67", "Pz", "E77", "E85", "E91", "T6",
        "E65", "E66", "E72", "E84", "E90", "O1", "Oz", "O2",
    ],
    91: HUMAN_EEG_ALL_CHANNELS_91,
}


HUMAN_EEG_2D_POSITIONS: dict[str, list[float]] = {
    "C3": [264.0, 776.0], "C4": [1335.0, 776.0], "Cz": [800.0, 804.0],
    "E10": [961.0, 34.0], "E101": [1344.0, 1127.0],
    "E102": [1431.0, 1002.0], "E103": [1437.0, 878.0],
    "E105": [1176.0, 706.0], "E106": [928.0, 640.0],
    "E109": [1434.0, 801.0], "E110": [1394.0, 665.0],
    "E111": [1263.0, 541.0], "E112": [1047.0, 472.0],
    "E115": [1367.0, 737.0], "E116": [1390.0, 547.0],
    "E117": [1312.0, 420.0], "E118": [1168.0, 311.0],
    "E12": [653.0, 240.0], "E123": [1297.0, 326.0],
    "E13": [552.0, 472.0], "E15": [800.0, 42.0],
    "E16": [800.0, 3.0], "E18": [638.0, 34.0],
    "E19": [532.0, 136.0], "E2": [1231.0, 257.0],
    "E20": [431.0, 311.0], "E23": [459.0, 131.0],
    "E26": [368.0, 257.0], "E27": [302.0, 326.0],
    "E28": [287.0, 420.0], "E29": [336.0, 541.0],
    "E3": [1140.0, 131.0], "E30": [423.0, 706.0],
    "E31": [594.0, 876.0], "E34": [209.0, 547.0],
    "E35": [205.0, 665.0], "E37": [399.0, 959.0],
    "E39": [232.0, 737.0], "E4": [1067.0, 136.0],
    "E40": [165.0, 801.0], "E41": [162.0, 878.0],
    "E42": [225.0, 1029.0], "E46": [168.0, 1002.0],
    "E47": [186.0, 1078.0], "E5": [946.0, 240.0],
    "E50": [255.0, 1127.0], "E51": [227.0, 1211.0],
    "E53": [382.0, 1206.0], "E54": [564.0, 1158.0],
    "E55": [800.0, 1030.0], "E59": [342.0, 1370.0],
    "E6": [800.0, 413.0], "E60": [410.0, 1377.0],
    "E61": [565.0, 1351.0], "E65": [423.0, 1394.0],
    "E66": [481.0, 1481.0], "E67": [626.0, 1496.0],
    "E7": [671.0, 640.0], "E71": [675.0, 1552.0],
    "E72": [800.0, 1528.0], "E76": [924.0, 1552.0],
    "E77": [973.0, 1496.0], "E78": [1034.0, 1351.0],
    "E79": [1035.0, 1158.0], "E80": [1005.0, 876.0],
    "E84": [1118.0, 1481.0], "E85": [1189.0, 1377.0],
    "E86": [1217.0, 1206.0], "E87": [1200.0, 959.0],
    "E90": [1176.0, 1394.0], "E91": [1257.0, 1370.0],
    "E93": [1374.0, 1029.0], "E97": [1372.0, 1211.0],
    "E98": [1413.0, 1078.0], "F3": [391.0, 252.0],
    "F4": [1208.0, 252.0], "F7": [287.0, 473.0],
    "F8": [1312.0, 473.0], "Fp1": [575.0, 64.0],
    "Fp2": [1024.0, 64.0], "Fz": [800.0, 59.0],
    "O1": [577.0, 1504.0], "O2": [1022.0, 1504.0],
    "Oz": [800.0, 1541.0], "P3": [255.0, 1223.0],
    "P4": [1344.0, 1223.0], "Pz": [800.0, 1458.0],
    "T3": [219.0, 952.0], "T4": [1380.0, 952.0],
    "T5": [311.0, 1269.0], "T6": [1288.0, 1269.0],
}


# =============================================================================
# Marmoset ECoG metadata
# =============================================================================

@dataclass(frozen=True)
class MarmosetRecordingInfo:
    date_prefix: str
    session_indices: tuple[int, ...]


MARMOSET_RECORDINGS: dict[str, MarmosetRecordingInfo] = {
    "Ji": MarmosetRecordingInfo("Ji20180308", (1, 3, 5, 15)),
    "Or": MarmosetRecordingInfo("Or20171207", (2, 4, 6, 16)),
    "Ji2": MarmosetRecordingInfo("Ji20181207", (4, None)),
    "Rc2": MarmosetRecordingInfo("Rc20181219", (8, None)),
}


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_HUMAN_WINDOW = slice(0, 2500)
DEFAULT_MARMOSET_WINDOW = slice(0, 25000)
DEFAULT_MARMOSET_SAMPLERATE = 1000.0


# =============================================================================
# Core dataclasses
# =============================================================================

@dataclass(frozen=True)
class NeuralDataset:
    """
    Common data format:
        data.shape = (n_trials, n_channels, n_times)

    Human EEG:
        n_trials = epochs

    Marmoset ECoG:
        n_trials = 1
    """

    name: str
    data: np.ndarray
    channel_names: list[str]
    samplerate: float
    mne_epochs: mne.Epochs | None = None

    @property
    def n_trials(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def n_times(self) -> int:
        return self.data.shape[2]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    feature: FeatureName
    band: Band | None = None


@dataclass(frozen=True)
class HumanEEGRecording:
    patient_id: int
    state_id: str
    file_stem: str

    @property
    def file_name(self) -> str:
        return f"{self.file_stem}{self.state_id}"

    def path(self, data_dir: Path = HUMAN_DATA_DIR) -> Path:
        return data_dir / f"{self.file_name}.set"


@dataclass(frozen=True)
class MarmosetECoGRecording:
    animal: Literal["Ji", "Or"]
    session_index: int
    date_prefix: str

    @property
    def session_dir_name(self) -> str:
        return f"{self.date_prefix}S{self.session_index}c"

    def session_dir(self, data_dir: Path = MARMOSET_DATA_DIR) -> Path:
        return data_dir / self.session_dir_name


# =============================================================================
# Metadata helpers
# =============================================================================

def get_human_recording(patient_id: int, state_id: str) -> HumanEEGRecording:
    if patient_id not in HUMAN_EEG_RECORDINGS:
        valid_ids = sorted(HUMAN_EEG_RECORDINGS)
        raise ValueError(
            f"Unknown patient_id={patient_id}. Valid patient IDs are: {valid_ids}"
        )

    info = HUMAN_EEG_RECORDINGS[patient_id]

    if state_id not in info.state_ids:
        raise ValueError(
            f"Unknown state_id='{state_id}' for patient_id={patient_id}. "
            f"Valid state IDs are: {list(info.state_ids)}"
        )

    return HumanEEGRecording(
        patient_id=patient_id,
        state_id=state_id,
        file_stem=info.file_stem,
    )


def get_human_channel_names(dim: int) -> list[str]:
    if dim not in HUMAN_EEG_CHANNEL_SETS:
        valid_dims = sorted(HUMAN_EEG_CHANNEL_SETS)
        raise ValueError(f"Unknown human EEG dim={dim}. Valid dims are: {valid_dims}")

    return HUMAN_EEG_CHANNEL_SETS[dim]


def get_marmoset_recording(
    animal: Literal["Ji", "Or"],
    session_index: int,
) -> MarmosetECoGRecording:
    if animal not in MARMOSET_RECORDINGS:
        valid_animals = sorted(MARMOSET_RECORDINGS)
        raise ValueError(f"Unknown animal='{animal}'. Valid animals are: {valid_animals}")

    info = MARMOSET_RECORDINGS[animal]

    if session_index not in info.session_indices:
        raise ValueError(
            f"Unknown session_index={session_index} for animal='{animal}'. "
            f"Valid sessions are: {list(info.session_indices)}"
        )

    return MarmosetECoGRecording(
        animal=animal,
        session_index=session_index,
        date_prefix=info.date_prefix,
    )


# =============================================================================
# Loaders
# =============================================================================

def load_human_eeg(
    patient_id: int,
    state_id: str,
    *,
    window: slice = DEFAULT_HUMAN_WINDOW,
    data_dir: Path = HUMAN_DATA_DIR,
) -> NeuralDataset:
    recording = get_human_recording(patient_id, state_id)
    path = recording.path(data_dir)

    epochs = mne.io.read_epochs_eeglab(
        path,
        verbose=False,
        montage_units="cm",
    )

    data = epochs.get_data()
    data = data[:, :, window]

    return NeuralDataset(
        name=f"human_eeg_patient{patient_id}_{state_id}",
        data=data,
        channel_names=list(epochs.ch_names),
        samplerate=float(epochs.info["sfreq"]),
        mne_epochs=epochs,
    )

from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io


def load_marmoset_ecog(
    animal: Literal["Ji", "Or"],
    session_index: int,
    *,
    window: slice = DEFAULT_MARMOSET_WINDOW,
    data_dir: Path = MARMOSET_DATA_DIR,
    samplerate: float = DEFAULT_MARMOSET_SAMPLERATE,
) -> NeuralDataset:
    recording = get_marmoset_recording(animal, session_index)
    session_dir = recording.session_dir(data_dir)

    channels = []

    for channel in range(1, 97):
        mat_path = session_dir / f"ECoG_ch{channel}.mat"
        # channel_data = scipy.io.loadmat(mat_path)["ECoGData"][:, window]
        channel_data = scipy.io.loadmat(mat_path)[f"ECoGData_ch{channel}"][:, window]
        channels.append(channel_data)

    # shape: (n_channels, n_times)
    data = np.concatenate(channels, axis=0)

    # shape: (n_channels, n_times) -> (1, n_channels, n_times)
    data = data[np.newaxis, :, :]

    return NeuralDataset(
        name=f"marmoset_ecog_{animal}_{session_index}",
        data=data,
        channel_names=[f"ECoG{channel}" for channel in range(1, 97)],
        samplerate=float(samplerate),
        mne_epochs=None,
    )


def _load_marmoset_event_samples(
    event_mat_path: Path,
    *,
    event_key: str = "cntEvent",
    event_sample_column: int = 5,
    event_time_unit: Literal["samples_matlab", "samples_python", "seconds"] = "samples_matlab",
    samplerate: float = DEFAULT_MARMOSET_SAMPLERATE,
) -> np.ndarray:
    """
    Event.mat からイベント onset のサンプル番号を読み込む。

    今回の Event.mat では:
        cntEvent shape = (1000, 6)

    6列目、つまり Python index では 5 列目にイベント時刻が入っている。

    Parameters
    ----------
    event_mat_path:
        Event.mat のパス。

    event_key:
        Event.mat 内のイベント変数名。
        今回は "cntEvent"。

    event_sample_column:
        イベント時刻が入っている列。
        Python の 0-based index で指定する。
        今回は 6列目なので 5。

    event_time_unit:
        "samples_matlab":
            MATLAB 由来の 1-based サンプル番号として扱い、Python用に -1 する。

        "samples_python":
            すでに Python 形式の 0-based サンプル番号として扱う。

        "seconds":
            秒単位のイベント時刻として扱い、samplerate を掛けてサンプル番号にする。

    Returns
    -------
    event_samples:
        Python 形式の 0-based サンプル番号。
    """

    event_mat = scipy.io.loadmat(event_mat_path)

    if event_key not in event_mat:
        available_keys = [k for k in event_mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"{event_mat_path} に {event_key!r} がありません。"
            f"利用可能な key は {available_keys} です。"
        )

    events = np.asarray(event_mat[event_key])

    if events.ndim == 1:
        event_values = events
    elif events.ndim == 2:
        if event_sample_column >= events.shape[1]:
            raise ValueError(
                f"{event_key!r} の shape は {events.shape} ですが、"
                f"event_sample_column={event_sample_column} が範囲外です。"
            )
        event_values = events[:, event_sample_column]
    else:
        raise ValueError(
            f"{event_key!r} は 1D または 2D を想定していますが、"
            f"shape={events.shape} でした。"
        )

    # NaN を除外
    event_values = np.asarray(event_values, dtype=float)
    event_values = event_values[~np.isnan(event_values)]

    if event_time_unit == "samples_matlab":
        event_samples = event_values.astype(int) - 1

    elif event_time_unit == "samples_python":
        event_samples = event_values.astype(int)

    elif event_time_unit == "seconds":
        event_samples = np.round(event_values * samplerate).astype(int)

    else:
        raise ValueError(f"Unknown event_time_unit: {event_time_unit}")

    return event_samples


def load_marmoset_ecog_epoched(
    animal: Literal["Ji", "Or"],
    session_index: int,
    *,
    epoch_window: tuple[float, float] = (-1.5, 1.5),
    window: slice = DEFAULT_MARMOSET_WINDOW,
    data_dir: Path = MARMOSET_DATA_DIR,
    samplerate: float = DEFAULT_MARMOSET_SAMPLERATE,
    event_filename: str = "Event.mat",
    event_key: str = "cntEvent",
    event_sample_column: int = 5,
    event_time_unit: Literal["samples_matlab", "samples_python", "seconds"] = "samples_matlab",
    drop_out_of_bounds: bool = True,
) -> NeuralDataset:
    """
    Event.mat の tone onset に合わせて ECoG を epoch 化して読み込む。

    元の load_marmoset_ecog は変更せず、別関数として定義する。

    Parameters
    ----------
    epoch_window:
        イベント onset を 0 秒とした epoch 範囲。

        例:
            (-1.5, 1.5)

        samplerate=1000 Hz の場合:
            -1500 ms 〜 +1500 ms
            合計 3000 samples

    window:
        元の連続データを読み込む範囲。
        DEFAULT_MARMOSET_WINDOW を使う場合、イベントサンプルから window.start を差し引いて補正する。

    event_filename:
        ECoG_ch*.mat と同じディレクトリにある Event.mat のファイル名。

    event_key:
        Event.mat 内のイベント変数名。
        今回の例では "cntEvent"。

    event_sample_column:
        イベント時刻が入っている列。
        今回の Event.mat では cntEvent の 6列目なので、Python index で 5。

    event_time_unit:
        今回の Event.mat ではサンプル番号に見えるため、通常は "samples_matlab" を使う。

    drop_out_of_bounds:
        True の場合、epoch がデータ範囲外にはみ出すイベントを捨てる。
        False の場合、範囲外を NaN padding する。

    Returns
    -------
    NeuralDataset:
        data shape = (n_events, n_channels, n_epoch_samples)

        epoch_window=(-1.5, 1.5), samplerate=1000 の場合:
            data shape = (n_events, 96, 3000)
    """

    recording = get_marmoset_recording(animal, session_index)
    session_dir = recording.session_dir(data_dir)

    # -------------------------
    # 1. Event.mat を読み込む
    # -------------------------
    event_mat_path = session_dir / event_filename

    event_samples = _load_marmoset_event_samples(
        event_mat_path,
        event_key=event_key,
        event_sample_column=event_sample_column,
        event_time_unit=event_time_unit,
        samplerate=samplerate,
    )

    # -------------------------
    # 2. ECoG を読み込む
    # -------------------------
    channels = []

    for channel in range(1, 97):
        mat_path = session_dir / f"ECoG_ch{channel}.mat"
        channel_data = scipy.io.loadmat(mat_path)["ECoGData"][:, window]
        channels.append(channel_data)

    # shape: (n_channels, n_times)
    continuous_data = np.concatenate(channels, axis=0)

    n_channels, n_times = continuous_data.shape

    # -------------------------
    # 3. window.start の分だけイベント時刻を補正
    # -------------------------
    window_start = 0 if window.start is None else window.start
    event_samples = event_samples - window_start

    # -------------------------
    # 4. epoch 範囲をサンプル数へ変換
    # -------------------------
    epoch_start_sec, epoch_end_sec = epoch_window

    start_offset = int(round(epoch_start_sec * samplerate))
    end_offset = int(round(epoch_end_sec * samplerate))

    n_epoch_samples = end_offset - start_offset

    if n_epoch_samples <= 0:
        raise ValueError(
            f"epoch_window={epoch_window} は不正です。"
            " epoch_window[1] は epoch_window[0] より大きくしてください。"
        )

    # 1 kHz, (-1.5, 1.5) なら 3000 samples
    expected_pre_samples = int(round(abs(epoch_start_sec) * samplerate))
    expected_post_samples = int(round(epoch_end_sec * samplerate))

    # -------------------------
    # 5. epoch 作成
    # -------------------------
    epochs = []
    kept_event_samples = []
    dropped_event_samples = []

    for event_sample in event_samples:
        start = event_sample + start_offset
        stop = event_sample + end_offset

        if start < 0 or stop > n_times:
            if drop_out_of_bounds:
                dropped_event_samples.append(event_sample)
                continue

            epoch = np.full(
                shape=(n_channels, n_epoch_samples),
                fill_value=np.nan,
                dtype=float,
            )

            src_start = max(start, 0)
            src_stop = min(stop, n_times)

            dst_start = src_start - start
            dst_stop = dst_start + (src_stop - src_start)

            if src_start < src_stop:
                epoch[:, dst_start:dst_stop] = continuous_data[:, src_start:src_stop]

        else:
            epoch = continuous_data[:, start:stop]

        if epoch.shape[-1] != n_epoch_samples:
            raise RuntimeError(
                f"epoch のサンプル数が不正です。"
                f" expected={n_epoch_samples}, got={epoch.shape[-1]}"
            )

        epochs.append(epoch)
        kept_event_samples.append(event_sample)

    if len(epochs) == 0:
        raise ValueError(
            "有効な epoch が 0 個でした。"
            " event_time_unit, event_sample_column, window, epoch_window を確認してください。"
        )

    # shape: (n_events, n_channels, n_epoch_samples)
    data = np.stack(epochs, axis=0)

    print(
        f"Created epochs: {data.shape[0]} events, "
        f"{data.shape[1]} channels, "
        f"{data.shape[2]} samples per epoch"
    )
    print(
        f"Epoch window: {epoch_window[0]} to {epoch_window[1]} sec "
        f"at {samplerate} Hz"
    )
    print(
        f"Pre samples: {expected_pre_samples}, "
        f"Post samples: {expected_post_samples}"
    )

    if len(dropped_event_samples) > 0:
        print(f"Dropped out-of-bounds events: {len(dropped_event_samples)}")

    return NeuralDataset(
        name=f"marmoset_ecog_{animal}_{session_index}_epoched",
        data=data,
        channel_names=[f"ECoG{channel}" for channel in range(1, 97)],
        samplerate=float(samplerate),
        mne_epochs=None,
    )


def split_marmoset_pre_post_1500ms(
    data: np.ndarray,
    *,
    samplerate: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    epoch_window=(-1.5, 1.5) で作ったデータを pre/post に分割する。

    Parameters
    ----------
    data:
        shape = (n_events, n_channels, n_times)

    Returns
    -------
    pre:
        -1500 ms 〜 0 ms
        shape = (n_events, n_channels, 1500)

    post:
        0 ms 〜 +1500 ms
        shape = (n_events, n_channels, 1500)
    """

    n_pre = int(round(1.5 * samplerate))
    n_post = int(round(1.5 * samplerate))
    required_samples = n_pre + n_post

    if data.shape[-1] < required_samples:
        raise ValueError(
            f"data.shape[-1]={data.shape[-1]} ですが、"
            f"pre/post 1500 ms ずつには {required_samples} samples 必要です。"
        )

    pre = data[:, :, :n_pre]
    post = data[:, :, n_pre:n_pre + n_post]

    return pre, post


# =============================================================================
# Signal processing
# =============================================================================

def bandpass(
    signal: np.ndarray,
    band: Band,
    *,
    samplerate: float,
) -> np.ndarray:
    return utils.get_bandpass(
        signal,
        start=band[0],
        end=band[1],
        samplerate=samplerate,
    )


def hilbert_feature(
    signal: np.ndarray,
    band: Band,
    feature: Literal["phase", "envelope"],
    *,
    samplerate: float,
    verbose: bool = False,
) -> np.ndarray:
    filtered = bandpass(
        signal,
        band,
        samplerate=samplerate,
    )

    _, envelope, phase, _ = utils.hilbert_transform(
        signal=filtered,
        verbose=verbose,
    )

    if feature == "phase":
        return phase

    if feature == "envelope":
        return envelope

    raise ValueError(f"Unsupported Hilbert feature: {feature}")


def transform_signal(
    signal: np.ndarray,
    spec: FeatureSpec,
    *,
    samplerate: float,
) -> np.ndarray:
    if spec.feature == "raw":
        return signal

    if spec.band is None:
        raise ValueError(f"`band` is required for feature='{spec.feature}'")

    if spec.feature == "band":
        return bandpass(
            signal,
            spec.band,
            samplerate=samplerate,
        )

    if spec.feature in {"phase", "envelope"}:
        return hilbert_feature(
            signal,
            spec.band,
            spec.feature,
            samplerate=samplerate,
        )

    raise ValueError(f"Unsupported feature: {spec.feature}")


def extract_feature_matrix(
    dataset: NeuralDataset,
    spec: FeatureSpec,
    *,
    trials: Iterable[int] | None = None,
    channels: Iterable[int] | None = None,
) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        shape = (n_selected_trials * n_times, n_selected_channels)
    """
    if trials is None:
        trials = range(dataset.n_trials)

    if channels is None:
        channels = range(dataset.n_channels)

    trial_matrices = []

    for trial_idx in trials:
        channel_features = []

        for channel_idx in channels:
            signal = dataset.data[trial_idx, channel_idx]
            feature_signal = transform_signal(
                signal,
                spec,
                samplerate=dataset.samplerate,
            )
            channel_features.append(feature_signal)

        trial_matrix = np.column_stack(channel_features)
        trial_matrices.append(trial_matrix)

    return np.concatenate(trial_matrices, axis=0)


# =============================================================================
# Channel selection
# =============================================================================

def channel_indices_from_names(
    available_channel_names: list[str],
    selected_channel_names: list[str],
) -> list[int]:
    name_to_index = {
        name: index for index, name in enumerate(available_channel_names)
    }

    missing = [
        name for name in selected_channel_names
        if name not in name_to_index
    ]

    if missing:
        raise ValueError(
            f"Selected channels are not found in the dataset: {missing}"
        )

    return [name_to_index[name] for name in selected_channel_names]


def human_channel_indices(dataset: NeuralDataset, dim: int) -> list[int]:
    selected_channel_names = get_human_channel_names(dim)

    return channel_indices_from_names(
        available_channel_names=dataset.channel_names,
        selected_channel_names=selected_channel_names,
    )


def select_channels_by_indices(
    data: np.ndarray,
    channel_indices: list[int],
    *,
    verbose: bool = True,
) -> np.ndarray:
    if verbose:
        print(
            "Index of selected electrodes(0~d)",
            len(channel_indices),
            channel_indices,
        )

    return data[:, channel_indices]


def select_human_channels(
    data: np.ndarray,
    dataset: NeuralDataset,
    dim: int,
    *,
    verbose: bool = True,
) -> np.ndarray:
    channel_indices = human_channel_indices(dataset, dim)

    return select_channels_by_indices(
        data,
        channel_indices,
        verbose=verbose,
    )
