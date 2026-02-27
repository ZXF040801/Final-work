"""
anomaly_root.py — Anomaly detection on root 0/ and 1/ folders (255 files total).
Lists every recording whose 15-feature profile contradicts its UPDRS label.
"""

import os
import sys
import numpy as np
from pathlib import Path

# ── preprocessing constants ─────────────────────────────────────────────────
FS        = 60
WIN_LEN   = 300   # 5s @ 60Hz
STRIDE    = 75    # 75% overlap
SENSOR_S1 = "01"
SENSOR_S2 = "02"

# ── feature names ──────────────────────────────────────────────────────────
FEAT_NAMES = [
    "dist_iti_mean", "dist_iti_cv", "dist_tap_rate", "dist_amp_mean",
    "dist_pow_0_2hz", "dist_pow_2_5hz",
    "s1_roll_pow_0_2hz", "s1_roll_pow_2_5hz",
    "s1_pitch_pow_0_2hz", "s1_pitch_pow_2_5hz",
    "s2_y_pow_0_2hz", "s2_y_pow_2_5hz",
    "s1_pitch_vel_rms", "s1_pitch_vel_std", "s1_pitch_vel_p95",
]

# Strong discriminative features (skip dist_iti_cv index=1 which is near-useless)
STRONG_IDX = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# Expected direction: UPDRS=0 is HIGHER for these features (positive means cls0 > cls1)
# Based on actual Cohen's d analysis
CLS0_HIGHER = {0: False,  # dist_iti_mean: cls0 LOW
               2: True,   # dist_tap_rate: cls0 HIGH
               3: True,   # dist_amp_mean: cls0 HIGH (tentative)
               4: False,  # dist_pow_0_2hz: cls0 LOW
               5: True,   # dist_pow_2_5hz: cls0 HIGH
               6: False,  # s1_roll_pow_0_2hz: cls0 LOW
               7: True,   # s1_roll_pow_2_5hz: cls0 HIGH
               8: False,  # s1_pitch_pow_0_2hz: cls0 LOW
               9: True,   # s1_pitch_pow_2_5hz: cls0 HIGH
               10: False, # s2_y_pow_0_2hz: cls0 LOW
               11: True,  # s2_y_pow_2_5hz: cls0 HIGH
               12: True,  # s1_pitch_vel_rms: cls0 HIGH
               13: True,  # s1_pitch_vel_std: cls0 HIGH
               14: True,  # s1_pitch_vel_p95: cls0 HIGH
               }

# ── signal processing ──────────────────────────────────────────────────────

def parse_raw_file(fpath):
    """Parse raw tapping file → two DataFrames (s1, s2) with t_ms, x, y, z, roll, pitch, yaw."""
    rows_s1, rows_s2 = [], []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            sid = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
                t_ms = float(parts[7])
            except ValueError:
                continue
            row = [t_ms, x, y, z, roll, pitch, yaw]
            if sid == SENSOR_S1:
                rows_s1.append(row)
            elif sid == SENSOR_S2:
                rows_s2.append(row)
    return rows_s1, rows_s2


def deduplicate(rows):
    seen = set()
    out = []
    for r in rows:
        t = r[0]
        if t not in seen:
            seen.add(t)
            out.append(r)
    return sorted(out, key=lambda r: r[0])


def resample_and_build_6ch(rows_s1, rows_s2):
    """Resample both sensors to FS Hz, return 6-channel array."""
    if len(rows_s1) < 4 or len(rows_s2) < 4:
        return None
    arr1 = np.array(rows_s1)  # [t, x, y, z, roll, pitch, yaw]
    arr2 = np.array(rows_s2)
    t_start = max(arr1[0, 0], arr2[0, 0])
    t_end   = min(arr1[-1, 0], arr2[-1, 0])
    if t_end - t_start < 3000:   # < 3s
        return None
    t_new = np.arange(t_start, t_end, 1000.0 / FS)
    # interpolate: dist = sqrt(x^2+y^2+z^2) from s1; angles from s1; s2_y from s2
    dist  = np.sqrt(arr1[:, 1]**2 + arr1[:, 2]**2 + arr1[:, 3]**2)
    dist_i  = np.interp(t_new, arr1[:, 0], dist)
    s1_pit  = np.interp(t_new, arr1[:, 0], arr1[:, 5])
    s1_roll = np.interp(t_new, arr1[:, 0], arr1[:, 4])
    s1_x    = np.interp(t_new, arr1[:, 0], arr1[:, 1])
    s1_z    = np.interp(t_new, arr1[:, 0], arr1[:, 3])
    s2_y    = np.interp(t_new, arr2[:, 0], arr2[:, 2])
    sig6 = np.stack([dist_i, s1_pit, s1_roll, s1_x, s1_z, s2_y], axis=1)
    return sig6


def lowpass_filter(sig, cutoff=20.0, order=4):
    from scipy.signal import butter, filtfilt
    nyq = FS / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, sig, axis=0)


def sliding_windows(sig, win_len=WIN_LEN, stride=STRIDE):
    wins = []
    n = len(sig)
    i = 0
    while i + win_len <= n:
        wins.append(sig[i:i+win_len])
        i += stride
    return wins


def band_power_ratio(sig_1d, fs=FS):
    n = len(sig_1d)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    psd = np.abs(np.fft.rfft(sig_1d))**2
    p_low  = psd[(freqs >= 0) & (freqs < 2)].sum()
    p_high = psd[(freqs >= 2) & (freqs < 5)].sum()
    total  = p_low + p_high + 1e-12
    return p_low / total, p_high / total


def extract_features(win):
    """win: (WIN_LEN, 6) → 15 features."""
    dist  = win[:, 0]
    s1_p  = win[:, 1]   # pitch
    s1_r  = win[:, 2]   # roll
    s2_y  = win[:, 5]

    # ── tapping rhythm (ITI-based on dist channel) ──
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(dist, height=np.percentile(dist, 60), distance=int(FS * 0.15))
    if len(peaks) >= 2:
        iti = np.diff(peaks) / FS
        iti_mean = float(np.mean(iti))
        iti_cv   = float(np.std(iti) / (np.mean(iti) + 1e-12))
        tap_rate = 1.0 / (iti_mean + 1e-12)
        amp_mean = float(np.mean(dist[peaks]))
    else:
        iti_mean = 1.0
        iti_cv   = 1.0
        tap_rate = 0.0
        amp_mean = float(np.mean(dist))

    # ── frequency band powers ──
    dist_l, dist_h   = band_power_ratio(dist)
    r_low,  r_high   = band_power_ratio(s1_r)
    p_low,  p_high   = band_power_ratio(s1_p)
    y_low,  y_high   = band_power_ratio(s2_y)

    # ── angular velocity ──
    vel = np.diff(s1_p) * FS
    rms = float(np.sqrt(np.mean(vel**2)))
    std = float(np.std(vel))
    p95 = float(np.percentile(np.abs(vel), 95))

    return [iti_mean, iti_cv, tap_rate, amp_mean,
            dist_l, dist_h, r_low, r_high, p_low, p_high,
            y_low, y_high, rms, std, p95]


def features_for_file(fpath):
    """Return per-window features (skip win=0). Returns None if too short."""
    rows_s1, rows_s2 = parse_raw_file(fpath)
    rows_s1 = deduplicate(rows_s1)
    rows_s2 = deduplicate(rows_s2)
    sig6 = resample_and_build_6ch(rows_s1, rows_s2)
    if sig6 is None:
        return None
    sig6 = lowpass_filter(sig6)
    wins = sliding_windows(sig6)
    if len(wins) < 2:
        return None
    feats = []
    for idx, w in enumerate(wins):
        if idx == 0:   # skip startup window
            continue
        feats.append(extract_features(w))
    if not feats:
        return None
    return np.array(feats)   # (n_wins, 15)


# ── main analysis ──────────────────────────────────────────────────────────

def scan_folder(folder, label):
    """Return list of (fname, median_features) for each valid file."""
    results = []
    files = sorted(Path(folder).glob("*.txt"))
    total = len(files)
    for i, fp in enumerate(files):
        if (i+1) % 20 == 0:
            print(f"  [{label}] {i+1}/{total} ...", flush=True)
        try:
            feats = features_for_file(str(fp))
            if feats is None or len(feats) == 0:
                continue
            med = np.median(feats, axis=0)
            results.append((fp.name, med))
        except Exception as e:
            print(f"  ERROR {fp.name}: {e}")
    return results


print("Scanning 0/ folder (147 files)...")
cls0_data = scan_folder("0", "cls0")
print(f"  Valid: {len(cls0_data)}")

print("Scanning 1/ folder (108 files)...")
cls1_data = scan_folder("1", "cls1")
print(f"  Valid: {len(cls1_data)}")

# Build feature matrix
all_names  = [("0", n, f) for n, f in cls0_data] + [("1", n, f) for n, f in cls1_data]
all_labels = [0]*len(cls0_data) + [1]*len(cls1_data)
X = np.array([f for _, _, f in all_names])  # (N, 15)

# Z-score normalize
mu  = X.mean(axis=0)
sig = X.std(axis=0) + 1e-12
Xz  = (X - mu) / sig

# Class centroids (strong features only)
idx0 = [i for i, l in enumerate(all_labels) if l == 0]
idx1 = [i for i, l in enumerate(all_labels) if l == 1]
c0 = Xz[idx0][:, STRONG_IDX].mean(axis=0)
c1 = Xz[idx1][:, STRONG_IDX].mean(axis=0)

print(f"\nTotal valid: {len(all_names)} ({len(cls0_data)} cls0, {len(cls1_data)} cls1)")

# Detect anomalies
anomalies = []
for i, (lbl_str, fname, _) in enumerate(all_names):
    true_label = all_labels[i]
    z_strong = Xz[i, STRONG_IDX]
    d_own   = np.linalg.norm(z_strong - (c0 if true_label==0 else c1))
    d_other = np.linalg.norm(z_strong - (c1 if true_label==0 else c0))
    severity = d_own - d_other   # positive = closer to wrong class

    # Count features on wrong side
    wrong_count = 0
    wrong_feats = []
    for fi, gi in enumerate(STRONG_IDX):
        fname_f = FEAT_NAMES[gi]
        z_val = Xz[i, gi]
        cls0_higher = CLS0_HIGHER[gi]
        # For cls0 recording: should be on cls0 side
        # cls0 side = higher z if cls0_higher=True, else lower z
        if true_label == 0:
            on_wrong_side = (z_val < 0 and cls0_higher) or (z_val > 0 and not cls0_higher)
        else:
            on_wrong_side = (z_val > 0 and cls0_higher) or (z_val < 0 and not cls0_higher)
        if on_wrong_side:
            wrong_count += 1
            wrong_feats.append(fname_f)

    anomalies.append({
        "folder": lbl_str,
        "fname": fname,
        "d_own": d_own,
        "d_other": d_other,
        "severity": severity,
        "wrong_count": wrong_count,
        "wrong_feats": wrong_feats,
        "is_anomaly": d_other < d_own,
    })

# Sort by severity descending
anomalies.sort(key=lambda x: -x["severity"])

# Print anomalous files (closer to wrong class)
true_anomalies = [a for a in anomalies if a["is_anomaly"]]
print(f"\n{'='*70}")
print(f"ANOMALOUS RECORDINGS: {len(true_anomalies)} total")
print(f"  (closer to opposite-class centroid than own class)")
print(f"{'='*70}")

# Group by folder
for lbl in ["0", "1"]:
    grp = [a for a in true_anomalies if a["folder"] == lbl]
    print(f"\n── Folder {lbl}/ — {len(grp)} anomalies ──────────────────────────")
    for a in grp:
        sev = f"{a['severity']:+.2f}"
        print(f"  {a['fname']}")
        print(f"    severity={sev}  wrong_feats={a['wrong_count']}/14  "
              f"d_own={a['d_own']:.2f}  d_other={a['d_other']:.2f}")
        print(f"    wrong: {', '.join(a['wrong_feats'][:6])}"
              + ("..." if len(a['wrong_feats']) > 6 else ""))

# Also save full list
print(f"\n{'='*70}")
print("FULL RANKING (all files, severity = d_own - d_other, higher = more anomalous)")
print(f"{'='*70}")
print(f"{'Folder':<6} {'Severity':>8} {'Wrong':>5} {'d_own':>6} {'d_other':>7}  Filename")
print("-"*90)
for a in anomalies:
    mark = " *** ANOMALY" if a["is_anomaly"] else ""
    print(f"  {a['folder']:<4} {a['severity']:>+8.3f} {a['wrong_count']:>5}/14 "
          f"{a['d_own']:>6.2f} {a['d_other']:>7.2f}  {a['fname']}{mark}")
