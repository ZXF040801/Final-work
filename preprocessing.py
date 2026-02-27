import os
import pickle
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.interpolate import interp1d


# ========================== CONFIGURATION ==================================

class Config:
    DATA_DIR = "data"
    OUTPUT_DIR = "preprocessed"

    FS = 60
    LOWPASS_CUTOFF = 20.0
    LOWPASS_ORDER = 4

    WINDOW_SEC = 5.0
    OVERLAP = 0.75
    WINDOW_LEN = int(WINDOW_SEC * FS)   # 300
    STRIDE = int(WINDOW_LEN * (1 - OVERLAP))  # 75

    TARGET_SCORES = [0, 1]

    # 6 VAE channels (back-tracked from top discriminative features)
    VAE_CHANNEL_NAMES = [
        'dist',       # ch0: finger distance
        's1_pitch',   # ch1: thumb pitch (rad, unwrapped)
        's1_roll',    # ch2: thumb roll (rad, unwrapped)
        's1_x',       # ch3: thumb x position
        's1_z',       # ch4: thumb z (vertical tap axis)
        's2_y',       # ch5: index finger y position
    ]
    N_VAE_CHANNELS = 6

    # Channel indices within the 6-ch VAE window
    CH_DIST     = 0
    CH_S1_PITCH = 1
    CH_S1_ROLL  = 2
    CH_S1_X     = 3
    CH_S1_Z     = 4
    CH_S2_Y     = 5

    # 15 clinical features — all extractable from 6 VAE channels
    CLINICAL_FEATURE_NAMES = [
        # Tap rhythm on distance (4 features)
        'dist_iti_mean',         # 1  |d|=0.858  mean inter-tap interval
        'dist_tap_rate',         # 2  |d|=0.817  taps per second
        'dist_iti_cv',           # 3  rhythm stability (std/mean of ITI)
        'dist_tap_jitter',       # 4  mean(|diff(ITI)|) / mean(ITI)
        # Frequency band ratios (8 features)
        's1_roll_pow_2_5hz',     # 5  |d|=0.757
        'dist_pow_0_2hz',        # 6  |d|=0.745
        'dist_pow_2_5hz',        # 7  |d|=0.744
        's1_x_pow_0_2hz',        # 8  |d|=0.739
        's1_roll_pow_0_2hz',     # 9  |d|=0.720
        's1_x_pow_2_5hz',        # 10 |d|=0.711
        's2_y_pow_0_2hz',        # 11 |d|=0.632
        's1_pitch_pow_0_2hz',    # 12 |d|=0.580
        # s1_pitch velocity (3 features)
        's1_pitch_vel_rms',      # 13 |d|=0.593
        's1_pitch_vel_std',      # 14 |d|=0.592
        's1_pitch_vel_p95',      # 15 |d|=0.581
    ]
    N_CLINICAL_FEATURES = 15


# ========================== PARSING ========================================

def parse_raw_file(filepath):
    """
    Parse raw .txt → per-sensor positions, orientations, timestamps.
    Format: sensor_id  x  y  z  ori_a  ori_b  ori_c  timestamp_ms
    """
    s1_pos, s2_pos = [], []
    s1_ori, s2_ori = [], []
    s1_ts, s2_ts = [], []

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                sid = parts[0]
                pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                ori = [float(parts[4]), float(parts[5]), float(parts[6])]
                ts = float(parts[7])
                if sid == '01':
                    s1_pos.append(pos); s1_ori.append(ori); s1_ts.append(ts)
                elif sid == '02':
                    s2_pos.append(pos); s2_ori.append(ori); s2_ts.append(ts)
            except ValueError:
                continue

    return (np.array(s1_pos), np.array(s2_pos),
            np.array(s1_ori), np.array(s2_ori),
            np.array(s1_ts), np.array(s2_ts))


# ========================== RESAMPLING =====================================

def deduplicate_timestamps(data, timestamps):
    ts_unique, inverse = np.unique(timestamps, return_inverse=True)
    if len(ts_unique) == len(timestamps):
        return data, timestamps
    data_dedup = np.zeros((len(ts_unique), data.shape[1]))
    counts = np.zeros(len(ts_unique))
    for i, idx in enumerate(inverse):
        data_dedup[idx] += data[i]
        counts[idx] += 1
    data_dedup /= counts[:, None]
    return data_dedup, ts_unique


def resample_and_build_6ch(s1_pos, s1_ori, s1_ts, s2_pos, s2_ori, s2_ts, fs):
    """
    Resample both sensors → build 6-channel VAE input.
    Returns: (N, 6) array of [dist, s1_pitch, s1_roll, s1_x, s1_z, s2_y]
             or None on failure.
    """
    # Deduplicate
    s1_pos, s1_ts_p = deduplicate_timestamps(s1_pos, s1_ts)
    s1_ori, s1_ts_o = deduplicate_timestamps(s1_ori, s1_ts)
    s2_pos, s2_ts_p = deduplicate_timestamps(s2_pos, s2_ts)

    # Common time range
    t_start = max(s1_ts_p[0], s2_ts_p[0])
    t_end = min(s1_ts_p[-1], s2_ts_p[-1])
    if t_end <= t_start:
        return None
    t_uniform = np.arange(t_start, t_end, 1000.0 / fs)
    if len(t_uniform) < 10:
        return None

    N = len(t_uniform)

    # Interpolate s1 position (x, y, z)
    s1_xyz = np.zeros((N, 3))
    for col in range(3):
        s1_xyz[:, col] = interp1d(s1_ts_p, s1_pos[:, col], kind='linear',
                                   fill_value='extrapolate')(t_uniform)

    # Interpolate s2 position (x, y, z)
    s2_xyz = np.zeros((N, 3))
    for col in range(3):
        s2_xyz[:, col] = interp1d(s2_ts_p, s2_pos[:, col], kind='linear',
                                   fill_value='extrapolate')(t_uniform)

    # ch0: dist = ||s1_pos - s2_pos||
    dist = np.sqrt(np.sum((s1_xyz - s2_xyz) ** 2, axis=1))

    # ch1: s1_pitch — ori column 1, deg → rad → unwrap → interp
    s1_pitch_unwrap = np.unwrap(np.deg2rad(s1_ori[:, 1]))
    s1_pitch = interp1d(s1_ts_o, s1_pitch_unwrap, kind='linear',
                         fill_value='extrapolate')(t_uniform)

    # ch2: s1_roll — ori column 0, deg → rad → unwrap → interp
    s1_roll_unwrap = np.unwrap(np.deg2rad(s1_ori[:, 0]))
    s1_roll = interp1d(s1_ts_o, s1_roll_unwrap, kind='linear',
                        fill_value='extrapolate')(t_uniform)

    # ch3: s1_x
    s1_x = s1_xyz[:, 0]

    # ch4: s1_z
    s1_z = s1_xyz[:, 2]

    # ch5: s2_y
    s2_y = s2_xyz[:, 1]

    data = np.column_stack([dist, s1_pitch, s1_roll, s1_x, s1_z, s2_y])
    return data


def lowpass_filter(signal_1d, cutoff, fs, order=4):
    nyq = fs / 2.0
    if cutoff >= nyq:
        return signal_1d
    sos = butter(order, cutoff / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, signal_1d)


def sliding_window(features, window_len, stride):
    N = features.shape[0]
    D = features.shape[1] if features.ndim > 1 else 1
    if features.ndim == 1:
        features = features[:, np.newaxis]
    windows = []
    for start in range(0, N - window_len + 1, stride):
        windows.append(features[start:start + window_len].copy())
    if not windows:
        return np.empty((0, window_len, D))
    return np.array(windows)


# ========================== FEATURE HELPERS ================================

def power_band_ratio(signal, fs, f_low, f_high):
    """Power in [f_low, f_high) / total power. Zero-mean before FFT."""
    sig = signal - np.mean(signal)
    N = len(sig)
    if N < 4:
        return 0.0
    Y = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    P = Y ** 2
    P_total = P.sum()
    if P_total < 1e-12:
        return 0.0
    mask = (freqs >= f_low) & (freqs < f_high)
    return P[mask].sum() / P_total


def detect_taps_on_signal(signal, fs):
    """
    Detect tapping peaks using moving-average smoothing + find_peaks.
    Returns: tap_times (sec), amps, ITI (sec). Empty arrays if <3 peaks.
    """
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(signal, kernel, mode='same')

    mu = np.mean(smoothed)
    sigma = np.std(smoothed)
    if sigma < 1e-8:
        return np.array([]), np.array([]), np.array([])

    peaks, _ = find_peaks(
        smoothed,
        height=mu + 0.3 * sigma,
        distance=10,
        prominence=0.3 * sigma
    )

    if len(peaks) < 3:
        return np.array([]), np.array([]), np.array([])

    tap_times = peaks / fs
    amps = smoothed[peaks]
    ITI = np.diff(tap_times)

    return tap_times, amps, ITI


# ========================== 15 CLINICAL FEATURES ===========================

def extract_window_features(window, fs=60):
    """
    Extract 15 clinical features from one window (T, 6).

    Channel layout:
        0: dist   1: s1_pitch   2: s1_roll   3: s1_x   4: s1_z   5: s2_y

    Returns: (15,) float32
    """
    C = Config
    dt = 1.0 / fs
    duration = len(window) * dt

    dist     = window[:, C.CH_DIST]
    s1_pitch = window[:, C.CH_S1_PITCH]
    s1_roll  = window[:, C.CH_S1_ROLL]
    s1_x     = window[:, C.CH_S1_X]
    s2_y     = window[:, C.CH_S2_Y]

    # ==== 1-4: Tap rhythm on distance ====
    tap_times, amps, ITI = detect_taps_on_signal(dist, fs)

    if len(ITI) > 0:
        dist_iti_mean   = np.mean(ITI)
        iti_std         = np.std(ITI)
        dist_iti_cv     = iti_std / (dist_iti_mean + 1e-8)
        dist_tap_jitter = np.mean(np.abs(np.diff(ITI))) / (dist_iti_mean + 1e-8) \
                          if len(ITI) > 1 else 0.0
    else:
        dist_iti_mean   = 0.0
        dist_iti_cv     = 0.0
        dist_tap_jitter = 0.0

    dist_tap_rate = len(tap_times) / duration if len(tap_times) >= 3 else 0.0

    # ==== 5-12: Frequency band power ratios ====
    s1_roll_pow_2_5  = power_band_ratio(s1_roll,  fs, 2, 5)
    dist_pow_0_2     = power_band_ratio(dist,     fs, 0, 2)
    dist_pow_2_5     = power_band_ratio(dist,     fs, 2, 5)
    s1_x_pow_0_2     = power_band_ratio(s1_x,     fs, 0, 2)
    s1_roll_pow_0_2  = power_band_ratio(s1_roll,  fs, 0, 2)
    s1_x_pow_2_5     = power_band_ratio(s1_x,     fs, 2, 5)
    s2_y_pow_0_2     = power_band_ratio(s2_y,     fs, 0, 2)
    s1_pitch_pow_0_2 = power_band_ratio(s1_pitch, fs, 0, 2)

    # ==== 13-15: s1_pitch velocity features ====
    vel = np.diff(s1_pitch) / dt
    s1_pitch_vel_rms = np.sqrt(np.mean(vel ** 2))
    s1_pitch_vel_std = np.std(vel)
    s1_pitch_vel_p95 = np.percentile(np.abs(vel), 95)

    return np.array([
        dist_iti_mean,         # 1
        dist_tap_rate,         # 2
        dist_iti_cv,           # 3
        dist_tap_jitter,       # 4
        s1_roll_pow_2_5,       # 5
        dist_pow_0_2,          # 6
        dist_pow_2_5,          # 7
        s1_x_pow_0_2,          # 8
        s1_roll_pow_0_2,       # 9
        s1_x_pow_2_5,          # 10
        s2_y_pow_0_2,          # 11
        s1_pitch_pow_0_2,      # 12
        s1_pitch_vel_rms,      # 13
        s1_pitch_vel_std,      # 14
        s1_pitch_vel_p95,      # 15
    ], dtype=np.float32)


def extract_features_batch(windows, dt=1.0/60):
    """(N, T, 6) → (N, 15)"""
    fs = 1.0 / dt
    features = np.array([extract_window_features(w, fs) for w in windows])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ========================== FOLDER-BASED LOADING ===========================

def load_file_list_from_folders(data_dir, target_scores):
    """
    Load file list from folder structure:
        data_dir/0/*.txt   → UPDRS score 0 (Non-PD)
        data_dir/1/*.txt   → UPDRS score 1 (PD)

    Patient ID extracted from filename:
        PD-Ruijin_<PatientID>_tapping<hand>_<date>.txt
    """
    records = []
    for score in target_scores:
        folder = os.path.join(data_dir, str(score))
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith('.txt'):
                continue
            # Extract patient ID from filename
            parts = fname.replace('.txt', '').split('_')
            if len(parts) >= 2:
                patient_id = parts[1]
            else:
                patient_id = fname

            # Extract hand from filename
            fn_lower = fname.lower()
            hand = 'left' if 'left' in fn_lower else (
                   'right' if 'right' in fn_lower else 'unknown')

            records.append({
                'filename': fname,
                'filepath': os.path.join(folder, fname),
                'score': score,
                'patient_id': patient_id,
                'hand': hand,
            })
    return records


# ========================== NORMALISATION ==================================

def compute_normalization_stats(X):
    """Compute per-channel mean/std from (N, T, C) array."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def normalize(X, mean, std):
    return (X - mean) / std


# ========================== PROCESS ONE FILE ===============================

def process_single_file(filepath, cfg):
    """
    Process one recording.
    Returns:
        windows_vae:  (M, 300, 6)  — 6-channel for VAE
        windows_feat: (M, 15)      — clinical features for classifier
    """
    s1_pos, s2_pos, s1_ori, s2_ori, s1_ts, s2_ts = parse_raw_file(filepath)
    if len(s1_pos) < 10 or len(s2_pos) < 10:
        return None, None

    # Resample → 6-channel: [dist, s1_pitch, s1_roll, s1_x, s1_z, s2_y]
    data = resample_and_build_6ch(s1_pos, s1_ori, s1_ts,
                                   s2_pos, s2_ori, s2_ts, cfg.FS)
    if data is None:
        return None, None

    # Lowpass filter on position-based channels
    # ch0=dist, ch3=s1_x, ch4=s1_z, ch5=s2_y (NOT orientation channels)
    for col in [0, 3, 4, 5]:
        data[:, col] = lowpass_filter(data[:, col], cfg.LOWPASS_CUTOFF,
                                       cfg.FS, cfg.LOWPASS_ORDER)

    # Sliding window → (M, 300, 6)
    windows = sliding_window(data, cfg.WINDOW_LEN, cfg.STRIDE)
    if len(windows) == 0:
        return None, None

    # Extract 15 clinical features from each 6-ch window
    feats = extract_features_batch(windows, dt=1.0 / cfg.FS)

    return windows, feats


# ========================== MAIN PIPELINE ==================================

def preprocess_dataset(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    records = load_file_list_from_folders(cfg.DATA_DIR, cfg.TARGET_SCORES)
    print(f"[Folders] {len(records)} files with scores in {cfg.TARGET_SCORES}")

    all_raw, all_feat, all_labels, all_pids, all_fids = [], [], [], [], []
    metadata = []
    skipped = 0

    for rec in records:
        filepath = rec['filepath']
        if not os.path.exists(filepath):
            skipped += 1
            continue
        w_raw, w_feat = process_single_file(filepath, cfg)
        if w_raw is None or len(w_raw) == 0:
            skipped += 1
            continue
        n_win = len(w_raw)
        all_raw.append(w_raw)
        all_feat.append(w_feat)
        all_labels.extend([rec['score']] * n_win)
        all_pids.extend([rec['patient_id']] * n_win)
        all_fids.extend([rec['filename']] * n_win)
        metadata.append({'filename': rec['filename'],
                         'patient_id': rec['patient_id'],
                         'hand': rec['hand'], 'score': rec['score'],
                         'n_windows': n_win})

    if not all_raw:
        print("[ERROR] No files processed!")
        return None, None, None, None, None, None

    X_raw = np.concatenate(all_raw, axis=0)
    X_feat = np.concatenate(all_feat, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    patient_ids = np.array(all_pids)
    file_ids = np.array(all_fids)

    print(f"\n[Done] Processed {len(metadata)} files, skipped {skipped}")
    print(f"  X_raw  = {X_raw.shape} (6-ch VAE input)")
    print(f"  X_feat = {X_feat.shape} (15 clinical features)")
    print(f"  Labels: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    print(f"  Patients: {len(set(patient_ids))}")
    print(f"\n  VAE channels: {cfg.VAE_CHANNEL_NAMES}")
    print(f"\n  Clinical features:")
    for i, fn in enumerate(cfg.CLINICAL_FEATURE_NAMES):
        col = X_feat[:, i]
        print(f"    [{i+1:2d}] {fn:25s}: mean={col.mean():.4f}, std={col.std():.4f}")

    output_path = os.path.join(cfg.OUTPUT_DIR, 'preprocessed_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({
            'X_raw': X_raw, 'X_feat': X_feat, 'y': y,
            'patient_ids': patient_ids, 'file_ids': file_ids,
            'vae_channel_names': cfg.VAE_CHANNEL_NAMES,
            'clinical_feature_names': cfg.CLINICAL_FEATURE_NAMES,
            'metadata': metadata,
        }, f)
    print(f"  Saved to: {output_path}")
    return X_raw, X_feat, y, patient_ids, file_ids, metadata


# ========================== PATIENT-AWARE SPLIT ============================

def patient_aware_split(X_raw, X_feat, y, patient_ids, file_ids=None,
                        test_ratio=0.2, seed=42):
    from collections import Counter
    np.random.seed(seed)

    patient_labels = {}
    for pid, label in zip(patient_ids, y):
        patient_labels.setdefault(pid, []).append(label)
    patient_majority = {pid: Counter(labels).most_common(1)[0][0]
                        for pid, labels in patient_labels.items()}

    patients_0 = [p for p, l in patient_majority.items() if l == 0]
    patients_1 = [p for p, l in patient_majority.items() if l == 1]
    np.random.shuffle(patients_0)
    np.random.shuffle(patients_1)

    n_test_0 = max(1, int(len(patients_0) * test_ratio))
    n_test_1 = max(1, int(len(patients_1) * test_ratio))
    test_patients = set(patients_0[:n_test_0]) | set(patients_1[:n_test_1])

    tr = np.array([pid not in test_patients for pid in patient_ids])
    te = ~tr

    result = {
        'X_raw_tr': X_raw[tr], 'X_raw_te': X_raw[te],
        'X_feat_tr': X_feat[tr], 'X_feat_te': X_feat[te],
        'y_tr': y[tr], 'y_te': y[te],
        'pid_tr': patient_ids[tr], 'pid_te': patient_ids[te],
    }
    if file_ids is not None:
        result['fid_tr'] = file_ids[tr]
        result['fid_te'] = file_ids[te]
    return result


if __name__ == "__main__":
    cfg = Config()
    result = preprocess_dataset(cfg)
    if result[0] is not None:
        print(f"\n{'='*50}")
        print(f"PREPROCESSING COMPLETE")
        print(f"  VAE input:        {result[0].shape}  (6-ch)")
        print(f"  Classifier input: {result[1].shape} (15 features)")
        print(f"{'='*50}")