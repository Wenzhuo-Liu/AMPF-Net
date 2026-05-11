from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Callable, Literal

import mne
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from braindecode.datasets import create_from_mne_raw, BaseConcatDataset
from braindecode.preprocessing import Preprocessor, preprocess
from sklearn.preprocessing import scale

# =============================================================================
# CL-Drive Dataset (preprocessed version)
# =============================================================================
# This script processes the CL-Drive dataset:
# Multimodal Brain-Computer Interface for In-Vehicle Driver Cognitive Load Measurement
# Dataset page: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/JJ2YZZ
# Reference paper: https://ieeexplore.ieee.org/document/10382455 (IEEE TITS 2024)
# =============================================================================

# =============================================================================
# Paths & global parameters
# =============================================================================

BASE_PATH = Path('./CL-Drive(preprocessed)')
EEG_BASE_PATH = BASE_PATH / 'EEG'
ECG_BASE_PATH = BASE_PATH / 'ECG'
EDA_BASE_PATH = BASE_PATH / 'EDA'
GAZE_BASE_PATH = BASE_PATH / 'Gaze'
LABELS_PATH = BASE_PATH / 'Labels'

PARTICIPANT_IDS = [f'{i:02d}' for i in range(1, 22)]
NUM_LEVELS = 9
EPOCH_DURATION = 10
TIMESTAMP_START = 120

EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
EEG_SFREQ = 256
EEG_WINDOW_SAMPLES = 2560
EEG_STRIDE_SAMPLES = 2560

ECG_CHANNELS = ['ECG LL-RA CAL', 'ECG LA-RA CAL', 'ECG Vx-RL CAL']
ECG_SFREQ = 512
ECG_WINDOW_SAMPLES = 5120
ECG_STRIDE_SAMPLES = 5120

EDA_CHANNELS = ['GSR Resistance CAL', 'GSR Conductance CAL', 'GSR Resistance CAL.1', 'GSR Conductance CAL.1']
EDA_SFREQ = 128
EDA_WINDOW_SAMPLES = 1280
EDA_STRIDE_SAMPLES = 1280

GAZE_CHANNELS = ['ET_PupilLeft', 'ET_PupilRight', 'Gaze Velocity', 'Gaze Acceleration']
GAZE_SFREQ = 50
GAZE_WINDOW_SAMPLES = 500
GAZE_STRIDE_SAMPLES = 500

OUTPUT_DIR = Path('./multimodal_dataset')
SAVE_DATASET = True
SAVE_PROCESSED_ARRAYS = True

PROCESSED_BUNDLE_NAME = 'four_modal_processed.npz'
FOUR_MODAL_MANIFEST_NAME = 'four_modal_manifest.json'
BUNDLE_SCHEMA_VERSION = 1

ANNOTATION_TO_TARGET_MAPPING: dict[str, int] = {"0": 0, "1": 1}

EEG_LOWCUT, EEG_HIGHCUT = 0.4, 75
EEG_NOTCH_FREQ = 60
EEG_NOTCH_Q = 30

ECG_LOWCUT, ECG_HIGHCUT = 5, 15

EDA_LOWCUT, EDA_HIGHCUT = 0.05, 3

RawFilter = Callable[[mne.io.RawArray], mne.io.RawArray]


def filter_eeg_raw(raw: mne.io.RawArray) -> mne.io.RawArray:
    raw = raw.copy()
    raw.filter(EEG_LOWCUT, EEG_HIGHCUT, method="iir", iir_params=dict(order=2, ftype="butter"), verbose=False)
    raw.notch_filter(freqs=EEG_NOTCH_FREQ, notch_widths=EEG_NOTCH_FREQ / EEG_NOTCH_Q, verbose=False)
    return raw


def filter_ecg_raw(raw: mne.io.RawArray) -> mne.io.RawArray:
    raw = raw.copy()
    raw.filter(ECG_LOWCUT, ECG_HIGHCUT, method="iir", iir_params=dict(order=2, ftype="butter"), picks="all", verbose=False)
    return raw


def filter_eda_raw(raw: mne.io.RawArray) -> mne.io.RawArray:
    raw = raw.copy()
    raw.filter(l_freq=EDA_LOWCUT, h_freq=EDA_HIGHCUT, method="iir", iir_params=dict(order=2, ftype="butter"), picks="all", verbose=False)
    return raw


# =============================================================================
# Per-modality configuration (single place for paths, channels, windowing, filters)
# =============================================================================


@dataclass(frozen=True)
class ModalitySpec:
    key: str
    data_root: Path
    csv_filename: str 
    channels: tuple[str, ...]
    sfreq: float
    window_samples: int
    stride_samples: int
    mne_ch_types: tuple[str, ...]
    filter_for_aligned_windows: RawFilter | None
    load_kind: Literal["standard", "eda_resample"] = "standard"

    def csv_path(self, participant_id: str, level: int) -> Path:
        return self.data_root / f'participant_ID_{participant_id}' / self.csv_filename.format(level=level)


def load_label_data(participant_id: str) -> dict:
    p = LABELS_PATH / f'participant_ID_{participant_id}_modified.csv'
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {i: df[f'lvl_{i}'].dropna().astype(int).tolist() for i in range(1, NUM_LEVELS + 1)}


def create_annotations(labels: list, duration: float) -> mne.Annotations:
    onset = np.arange(0, duration * len(labels), duration)
    desc = [str(int(x)) for x in labels]
    for d in desc:
        if d not in ANNOTATION_TO_TARGET_MAPPING:
            raise ValueError(f"Description {d!r} missing from ANNOTATION_TO_TARGET_MAPPING.")
    return mne.Annotations(onset=onset, duration=duration, description=desc)


def align_timestamps(df: pd.DataFrame, data_columns: list) -> tuple[np.ndarray, np.ndarray]:
    ts_col = 'Timestamp' if 'Timestamp' in df.columns else df.columns[0]
    timestamps = df[ts_col].values.astype(float)
    data = df[data_columns].values.astype(float)

    mask = timestamps >= TIMESTAMP_START
    ts_valid = timestamps[mask]
    data_valid = data[mask]

    if len(ts_valid) == 0:
        return np.array([]), np.array([])

    t_start = ts_valid[0]
    if t_start > TIMESTAMP_START:
        dt = np.median(np.diff(ts_valid)) if len(ts_valid) > 1 else 0.004
        n_prepend = int(round((t_start - TIMESTAMP_START) / dt))
        if n_prepend > 0:
            prepend_data = np.full((n_prepend, data.shape[1]), np.nan, dtype=float)
            data_valid = np.vstack([prepend_data, data_valid])
            ts_prepend = np.linspace(TIMESTAMP_START, t_start - dt, n_prepend, endpoint=False)
            ts_valid = np.concatenate([ts_prepend, ts_valid])

    return ts_valid, data_valid


MODALITY_SPECS: dict[str, ModalitySpec] = {
    "eeg": ModalitySpec(
        key="eeg",
        data_root=EEG_BASE_PATH,
        csv_filename="eeg_data_level_{level}.csv",
        channels=tuple(EEG_CHANNELS),
        sfreq=EEG_SFREQ,
        window_samples=EEG_WINDOW_SAMPLES,
        stride_samples=EEG_STRIDE_SAMPLES,
        mne_ch_types=tuple(["eeg"] * len(EEG_CHANNELS)),
        filter_for_aligned_windows=filter_eeg_raw,
    ),
    "ecg": ModalitySpec(
        key="ecg",
        data_root=ECG_BASE_PATH,
        csv_filename="ecg_data_level_{level}_processed.csv",
        channels=tuple(ECG_CHANNELS),
        sfreq=ECG_SFREQ,
        window_samples=ECG_WINDOW_SAMPLES,
        stride_samples=ECG_STRIDE_SAMPLES,
        mne_ch_types=tuple(["ecg"] * len(ECG_CHANNELS)),
        filter_for_aligned_windows=filter_ecg_raw,
    ),
    "eda": ModalitySpec(
        key="eda",
        data_root=EDA_BASE_PATH,
        csv_filename="eda_data_level_{level}_processed.csv",
        channels=tuple(EDA_CHANNELS),
        sfreq=EDA_SFREQ,
        window_samples=EDA_WINDOW_SAMPLES,
        stride_samples=EDA_STRIDE_SAMPLES,
        mne_ch_types=tuple(["eog"] * len(EDA_CHANNELS)),
        filter_for_aligned_windows=filter_eda_raw,
        load_kind="eda_resample",
    ),
    "gaze": ModalitySpec(
        key="gaze",
        data_root=GAZE_BASE_PATH,
        csv_filename="gaze_data_level_{level}_processed.csv",
        channels=tuple(GAZE_CHANNELS),
        sfreq=GAZE_SFREQ,
        window_samples=GAZE_WINDOW_SAMPLES,
        stride_samples=GAZE_STRIDE_SAMPLES,
        mne_ch_types=tuple(["misc"] * len(GAZE_CHANNELS)),
        filter_for_aligned_windows=None,
    ),
}

FOUR_MODALITY_KEYS = ("eeg", "ecg", "eda", "gaze")


def load_modality_raw(participant_id: str, level: int, spec: ModalitySpec) -> mne.io.RawArray | None:
    path = spec.csv_path(participant_id, level)
    if not path.exists():
        return None
    df = pd.read_csv(path)

    if spec.load_kind == "eda_resample":
        eda_cols = [c for c in spec.channels if c in df.columns]
        if not eda_cols:
            return None
        ts_aligned, data_aligned = align_timestamps(df, eda_cols)
        if len(ts_aligned) == 0:
            return None
        t_target = np.arange(TIMESTAMP_START, ts_aligned[-1] + 1 / spec.sfreq, 1 / spec.sfreq)
        data_resampled = np.zeros((len(eda_cols), len(t_target)))
        for c in range(data_aligned.shape[1]):
            f = interp1d(ts_aligned, data_aligned[:, c], kind='linear', bounds_error=False, fill_value=0)
            data_resampled[c] = f(t_target)
        info = mne.create_info(eda_cols, spec.sfreq, ch_types=['eog'] * len(eda_cols))
        return mne.io.RawArray(data_resampled.astype(np.float64), info)

    ts_aligned, data_aligned = align_timestamps(df, list(spec.channels))
    if len(ts_aligned) == 0:
        return None
    data = data_aligned.T
    info = mne.create_info(list(spec.channels), spec.sfreq, ch_types=list(spec.mne_ch_types))
    return mne.io.RawArray(data, info)


# =============================================================================
# Braindecode: one continuous recording per level → sliding windows
# =============================================================================


def build_windowed_dataset(
    participant_ids: list[str],
    spec: ModalitySpec,
    *,
    apply_filter_to_continuous: bool = False,
    apply_normalization: bool = True,
) -> BaseConcatDataset | None:
    all_datasets = []
    for pid in participant_ids:
        label_data = load_label_data(pid)
        raws, desc = [], []
        for level in range(1, NUM_LEVELS + 1):
            if level not in label_data or not label_data[level]:
                continue
            raw = load_modality_raw(pid, level, spec)
            if raw is None:
                continue
            if apply_filter_to_continuous:
                if spec.filter_for_aligned_windows is not None:
                    raw = spec.filter_for_aligned_windows(raw)
            labels = [int(x) for x in label_data[level]]
            raw.set_annotations(create_annotations(labels, EPOCH_DURATION))
            raws.append(raw)
            desc.append({"subject": int(pid), "level": level})
        if not raws:
            continue
        ds = create_from_mne_raw(
            raws,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            window_size_samples=spec.window_samples,
            window_stride_samples=spec.stride_samples,
            drop_last_window=False,
            drop_bad_windows=False,
            accepted_bads_ratio=0.3,
            descriptions=desc,
            mapping=ANNOTATION_TO_TARGET_MAPPING,
        )
        all_datasets.append(ds)
    if not all_datasets:
        return None
    concat = all_datasets[0] if len(all_datasets) == 1 else BaseConcatDataset(all_datasets)
    if apply_normalization:
        preprocess(concat, [Preprocessor(scale, channel_wise=True, picks="all")], overwrite=True, n_jobs=1)
    return concat


def build_modality_key_to_index(dataset: BaseConcatDataset) -> dict:
    meta = dataset.get_metadata()
    subj_col = next((c for c in ('subject', 'subject_id', 'i_subject') if c in meta.columns), None)
    if subj_col is None:
        raise KeyError("No subject column in metadata.")
    if 'level' not in meta.columns:
        raise KeyError("No level column in metadata.")
    meta = meta.copy()
    meta["epoch_index"] = meta.groupby([subj_col, "level"]).cumcount()
    key_to_idx = {}
    for i in range(len(meta)):
        row = meta.iloc[i]
        key = (int(row[subj_col]), int(row["level"]), int(row["epoch_index"]))
        key_to_idx[key] = i
    return key_to_idx


# =============================================================================
# Intersect four modalities by (subject, level, epoch_index); drop NaN windows
# =============================================================================


def get_four_modality_aligned(
    raw_by_modality: dict[str, BaseConcatDataset],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    """Keys: (subject, level, epoch_index). Returns parallel lists of index quads and keys."""
    k2i = {m: build_modality_key_to_index(raw_by_modality[m]) for m in FOUR_MODALITY_KEYS}
    all_keys = set(k2i["eeg"])
    for m in FOUR_MODALITY_KEYS[1:]:
        all_keys &= set(k2i[m])

    def to_np(x):
        return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)

    indices = []
    keys_ok = []
    for k in sorted(all_keys):
        idxs = tuple(k2i[m][k] for m in FOUR_MODALITY_KEYS)
        Xs = tuple(to_np(raw_by_modality[m][idxs[i]][0]) for i, m in enumerate(FOUR_MODALITY_KEYS))
        if any(x.size == 0 or np.any(np.isnan(x)) for x in Xs):
            continue
        if Xs[2].shape[0] < len(EDA_CHANNELS):
            continue
        indices.append(idxs)
        keys_ok.append(k)
    return indices, keys_ok


# =============================================================================
# Aligned subset: optional filter in a synthetic Raw, then channel-wise normalize
# =============================================================================


def _filter_window(X: np.ndarray, ch_names: list, sfreq: float, ch_types: list, filter_fn: RawFilter) -> np.ndarray:
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(X.astype(np.float64), info)
    raw = filter_fn(raw)
    return raw.get_data()


def _normalize_window(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std_safe = np.where(std < eps, 1.0, std)
    return ((X - mean) / std_safe).astype(np.float32)


def build_processed_subset(
    dataset: BaseConcatDataset,
    indices: list[int],
    filter_fn: RawFilter | None,
    ch_names: list,
    sfreq: float,
    ch_types: list,
    apply_normalization: bool,
) -> ProcessedSubsetDataset:
    X_list, y_list = [], []
    for idx in indices:
        sample = dataset[idx]
        X = sample[0].numpy() if hasattr(sample[0], 'numpy') else np.asarray(sample[0])
        y = sample[1].item() if hasattr(sample[1], 'item') else int(sample[1])
        if filter_fn is not None:
            X = _filter_window(X, ch_names, sfreq, ch_types, filter_fn)
        if apply_normalization:
            X = _normalize_window(X)
        X_list.append(X)
        y_list.append(y)
    return ProcessedSubsetDataset(X_list, y_list)


# =============================================================================
# PyTorch datasets
# =============================================================================


class ProcessedSubsetDataset(Dataset):
    def __init__(self, X_list: list, y_list: list):
        self.X_list = X_list
        self.y_list = y_list

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, i):
        return np.asarray(self.X_list[i], dtype=np.float32), self.y_list[i]


class StackedProcessedModalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        if len(self.X) != len(self.y):
            raise ValueError("X and y length mismatch.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], int(self.y[i])


class FourModalDataset(Dataset):
    def __init__(self, eeg_ds, ecg_ds, eda_ds, gaze_ds):
        self.eeg_ds = eeg_ds
        self.ecg_ds = ecg_ds
        self.eda_ds = eda_ds
        self.gaze_ds = gaze_ds

    def __len__(self):
        return len(self.eeg_ds)

    def __getitem__(self, i):
        return (
            self.eeg_ds[i][0],
            self.ecg_ds[i][0],
            self.eda_ds[i][0],
            self.gaze_ds[i][0],
        ), self.eeg_ds[i][1]


# =============================================================================
# NPZ + manifest I/O
# =============================================================================


def bundle_paths_exist(directory: Path) -> bool:
    d = Path(directory)
    return (d / FOUR_MODAL_MANIFEST_NAME).is_file() and (d / PROCESSED_BUNDLE_NAME).is_file()


def save_four_modal_processed_bundle(
    output_dir: Path,
    eeg_dataset: ProcessedSubsetDataset,
    ecg_dataset: ProcessedSubsetDataset,
    eda_dataset: ProcessedSubsetDataset,
    gaze_dataset: ProcessedSubsetDataset,
    aligned_keys: list,
    participant_ids: list[str],
    extra_manifest: dict | None = None,
) -> tuple[Path, Path]:
    n = len(eeg_dataset)
    for name, ds in (
        ('ecg', ecg_dataset),
        ('eda', eda_dataset),
        ('gaze', gaze_dataset),
    ):
        if len(ds) != n:
            raise ValueError(f"Modal length mismatch: eeg={n}, {name}={len(ds)}")

    def stack_modal(ds: ProcessedSubsetDataset) -> np.ndarray:
        return np.stack(
            [np.asarray(ds.X_list[i], dtype=np.float32) for i in range(n)],
            axis=0,
        )

    eeg_arr = stack_modal(eeg_dataset)
    ecg_arr = stack_modal(ecg_dataset)
    eda_arr = stack_modal(eda_dataset)
    gaze_arr = stack_modal(gaze_dataset)
    labels = np.array(eeg_dataset.y_list, dtype=np.int64)
    for i in range(min(100, n)):
        if (
            ecg_dataset.y_list[i] != eeg_dataset.y_list[i]
            or eda_dataset.y_list[i] != eeg_dataset.y_list[i]
            or gaze_dataset.y_list[i] != eeg_dataset.y_list[i]
        ):
            raise ValueError(f"Label mismatch at index {i}.")

    keys_arr = np.array(aligned_keys, dtype=np.int64)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / PROCESSED_BUNDLE_NAME

    np.savez_compressed(
        bundle_path,
        eeg=eeg_arr,
        ecg=ecg_arr,
        eda=eda_arr,
        gaze=gaze_arr,
        labels=labels,
        aligned_keys=keys_arr,
    )

    manifest: dict = {
        'schema_version': BUNDLE_SCHEMA_VERSION,
        'bundle_filename': PROCESSED_BUNDLE_NAME,
        'n_samples': int(n),
        'participant_ids': list(participant_ids),
        'shapes': {
            'eeg': list(eeg_arr.shape),
            'ecg': list(ecg_arr.shape),
            'eda': list(eda_arr.shape),
            'gaze': list(gaze_arr.shape),
        },
        'dtype': 'float32',
        'labels_dtype': 'int64',
        'aligned_keys_shape': list(keys_arr.shape),
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    manifest_path = out_dir / FOUR_MODAL_MANIFEST_NAME
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved processed bundle: {bundle_path}")
    print(f"Manifest: {manifest_path}")
    return bundle_path, manifest_path


def load_four_modal_processed_bundle(
    load_dir: Path,
) -> tuple[FourModalDataset, StackedProcessedModalDataset, StackedProcessedModalDataset, StackedProcessedModalDataset, StackedProcessedModalDataset, list[tuple]]:
    load_dir = Path(load_dir)
    manifest_path = load_dir / FOUR_MODAL_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    schema = manifest.get('schema_version', 0)
    if schema != BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version={schema!r}, expected {BUNDLE_SCHEMA_VERSION}")

    bundle_name = manifest.get('bundle_filename', PROCESSED_BUNDLE_NAME)
    bundle_path = load_dir / bundle_name
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Missing NPZ bundle: {bundle_path}")

    z = np.load(bundle_path, allow_pickle=False)
    labels = np.asarray(z['labels'], dtype=np.int64)
    aligned_keys_arr = np.asarray(z['aligned_keys'], dtype=np.int64)
    aligned_keys = [tuple(int(x) for x in row) for row in aligned_keys_arr]

    eeg_ds = StackedProcessedModalDataset(z['eeg'], labels)
    ecg_ds = StackedProcessedModalDataset(z['ecg'], labels)
    eda_ds = StackedProcessedModalDataset(z['eda'], labels)
    gaze_ds = StackedProcessedModalDataset(z['gaze'], labels)
    multimodal = FourModalDataset(eeg_ds, ecg_ds, eda_ds, gaze_ds)

    n_manifest = manifest.get('n_samples')
    if n_manifest is not None and int(n_manifest) != len(multimodal):
        raise ValueError(f"manifest n_samples={n_manifest} != loaded {len(multimodal)}")

    print(f"Loaded four-modal bundle: {bundle_path} ({len(multimodal)} samples, schema={schema})")
    return multimodal, eeg_ds, ecg_ds, eda_ds, gaze_ds, aligned_keys


# =============================================================================
# CLI pipeline
# =============================================================================


def main(participant_ids: list[str] | None = None):
    if participant_ids is None:
        participant_ids = PARTICIPANT_IDS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60 + "\nFour-modal windows (EEG+ECG+EDA+Gaze)\n" + "=" * 60)
    print(f"Participants: {participant_ids}")

    raw_by_modality: dict[str, BaseConcatDataset] = {}
    for i, key in enumerate(FOUR_MODALITY_KEYS, start=1):
        spec = MODALITY_SPECS[key]
        print(f"\n[{i}/6] {key.upper()} raw windows...")
        ds = build_windowed_dataset(
            participant_ids,
            spec,
            apply_filter_to_continuous=False,
            apply_normalization=False,
        )
        if ds is None:
            raise RuntimeError(f"{key} build failed")
        raw_by_modality[key] = ds
        print(f"      {key.upper()} count: {len(ds)}")

    eeg_raw = raw_by_modality["eeg"]
    ecg_raw = raw_by_modality["ecg"]
    eda_raw = raw_by_modality["eda"]
    gaze_raw = raw_by_modality["gaze"]

    print("\n[5/6] Four-modal intersection...")
    aligned_indices, aligned_keys = get_four_modality_aligned(raw_by_modality)
    n_aligned = len(aligned_keys)
    print(f"      Aligned: {n_aligned}")
    print(
        "      Dropped EEG/ECG/EDA/Gaze: "
        f"{len(eeg_raw) - n_aligned}, {len(ecg_raw) - n_aligned}, {len(eda_raw) - n_aligned}, {len(gaze_raw) - n_aligned}"
    )

    print("\n[6/6] Filter + normalize on aligned subset...")
    processed: dict[str, ProcessedSubsetDataset] = {}
    for key in FOUR_MODALITY_KEYS:
        spec = MODALITY_SPECS[key]
        idx_col = FOUR_MODALITY_KEYS.index(key)
        modality_indices = [p[idx_col] for p in aligned_indices]
        fn = spec.filter_for_aligned_windows
        processed[key] = build_processed_subset(
            raw_by_modality[key],
            modality_indices,
            fn,
            list(spec.channels),
            spec.sfreq,
            list(spec.mne_ch_types),
            apply_normalization=True,
        )

    eeg_dataset = processed["eeg"]
    ecg_dataset = processed["ecg"]
    eda_dataset = processed["eda"]
    gaze_dataset = processed["gaze"]

    multimodal_dataset = FourModalDataset(eeg_dataset, ecg_dataset, eda_dataset, gaze_dataset)
    print(f"      Four-modal dataset size: {len(multimodal_dataset)}")

    info = {
        "eeg_samples_raw": len(eeg_raw),
        "ecg_samples_raw": len(ecg_raw),
        "eda_samples_raw": len(eda_raw),
        "gaze_samples_raw": len(gaze_raw),
        "aligned_samples": len(multimodal_dataset),
        "eeg_shape": list(eeg_dataset[0][0].shape),
        "ecg_shape": list(ecg_dataset[0][0].shape),
        "eda_shape": list(eda_dataset[0][0].shape),
        "gaze_shape": list(gaze_dataset[0][0].shape),
        "aligned_keys_sample": aligned_keys[:10],
        "participants": participant_ids,
        "preprocessing_order": "intersect keys first, then per-modality filter and normalize",
        "preprocessing": {
            "eeg": "bandpass 0.4-75 Hz, notch 60 Hz Q=30, channel-wise normalization",
            "ecg": "raw CSV cleanup + 512 Hz polynomial interpolation (order=5), then bandpass 5-15 Hz + channel-wise normalization",
            "eda": "raw CSV interpolation (order=5), then resample to 128 Hz + bandpass 0.05-3 Hz + channel-wise normalization",
            "gaze": "raw CSV blink/missing handling with sample-and-hold, then channel-wise normalization",
        },
    }
    info_path = OUTPUT_DIR / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"\n      dataset_info.json: {info_path}")

    if SAVE_DATASET:
        pairs_path = OUTPUT_DIR / 'aligned_keys.json'
        with open(pairs_path, 'w') as f:
            json.dump([list(k) for k in aligned_keys], f)
        print(f"      aligned_keys.json: {pairs_path}")

    if SAVE_PROCESSED_ARRAYS:
        extra = {
            'source': 'cl_drive_dataset_preprocess.main',
            'preprocessing_note': info.get('preprocessing_order', ''),
        }
        save_four_modal_processed_bundle(
            OUTPUT_DIR,
            eeg_dataset,
            ecg_dataset,
            eda_dataset,
            gaze_dataset,
            aligned_keys,
            participant_ids,
            extra_manifest=extra,
        )

    print("\n" + "-" * 60 + "\nShapes\n" + "-" * 60)
    print(f"EEG:  {eeg_dataset[0][0].shape}, {EEG_SFREQ} Hz")
    print(f"ECG:  {ecg_dataset[0][0].shape}, {ECG_SFREQ} Hz")
    print(f"EDA:  {eda_dataset[0][0].shape}, {EDA_SFREQ} Hz")
    print(f"Gaze: {gaze_dataset[0][0].shape}, {GAZE_SFREQ} Hz")
    print(f"__getitem__: ((X_eeg, X_ecg, X_eda, X_gaze), y)")
    print("=" * 60 + "\n")

    return multimodal_dataset, eeg_dataset, ecg_dataset, eda_dataset, gaze_dataset, aligned_keys


if __name__ == "__main__":
    main(participant_ids=PARTICIPANT_IDS)
