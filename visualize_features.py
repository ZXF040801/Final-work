"""
visualize_features.py
=====================
Comprehensive visualisation of the 15 clinical features extracted from
data/0/ (UPDRS=0) and data/1/ (UPDRS=1).

Panels produced
---------------
Fig 1  — Raw 6-channel signals: sample from class 0 vs class 1
Fig 2  — FFT periodicity analysis of 'dist' for multiple recordings
Fig 3  — Per-feature boxplots (15 features × 2 classes)
Fig 4  — Correlation matrix of 15 features (all windows)
Fig 5  — Feature distribution histograms with KDE (15 features)
Fig 6  — Class-0 vs Class-1 mean feature bar chart with error bars
Fig 7  — Per-recording dominant tap period distribution
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
OUT_DIR    = "results/viz"
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants (mirror preprocessing.py) ────────────────────────────────────
FS           = 60
LOWPASS_CUT  = 20.0
LOWPASS_ORD  = 4
WINDOW_SEC   = 5.0
OVERLAP      = 0.75
WINDOW_LEN   = int(WINDOW_SEC * FS)   # 300 samples
STRIDE       = int(WINDOW_LEN * (1 - OVERLAP))  # 75 samples

VAE_CH_NAMES = ['dist', 's1_pitch', 's1_roll', 's1_x', 's1_z', 's2_y']

FEAT_NAMES = [
    'dist_iti_mean',
    'dist_tap_rate',
    'dist_iti_cv',
    'dist_tap_jitter',
    's1_roll_pow_2_5hz',
    'dist_pow_0_2hz',
    'dist_pow_2_5hz',
    's1_x_pow_0_2hz',
    's1_roll_pow_0_2hz',
    's1_x_pow_2_5hz',
    's2_y_pow_0_2hz',
    's1_pitch_pow_0_2hz',
    's1_pitch_vel_rms',
    's1_pitch_vel_std',
    's1_pitch_vel_p95',
]

# ════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL HELPERS (mirror preprocessing.py)
# ════════════════════════════════════════════════════════════════════════════

def parse_raw_file(filepath):
    s1_pos, s2_pos = [], []
    s1_ori, s2_ori = [], []
    s1_ts,  s2_ts  = [], []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                sid = parts[0]
                pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                ori = [float(parts[4]), float(parts[5]), float(parts[6])]
                ts  = float(parts[7])
                if sid == '01':
                    s1_pos.append(pos); s1_ori.append(ori); s1_ts.append(ts)
                elif sid == '02':
                    s2_pos.append(pos); s2_ori.append(ori); s2_ts.append(ts)
            except ValueError:
                continue
    return (np.array(s1_pos), np.array(s2_pos),
            np.array(s1_ori), np.array(s2_ori),
            np.array(s1_ts),  np.array(s2_ts))


def dedup(data, ts):
    ts_u, inv = np.unique(ts, return_inverse=True)
    if len(ts_u) == len(ts):
        return data, ts
    d2 = np.zeros((len(ts_u), data.shape[1]))
    c  = np.zeros(len(ts_u))
    for i, idx in enumerate(inv):
        d2[idx] += data[i]; c[idx] += 1
    return d2 / c[:, None], ts_u


def build_6ch(s1_pos, s1_ori, s1_ts, s2_pos, s2_ori, s2_ts, fs=FS):
    s1_pos, s1_ts_p = dedup(s1_pos, s1_ts)
    s1_ori, s1_ts_o = dedup(s1_ori, s1_ts)
    s2_pos, s2_ts_p = dedup(s2_pos, s2_ts)

    t0 = max(s1_ts_p[0], s2_ts_p[0])
    t1 = min(s1_ts_p[-1], s2_ts_p[-1])
    if t1 <= t0:
        return None
    t_uni = np.arange(t0, t1, 1000.0 / fs)
    if len(t_uni) < 10:
        return None

    N = len(t_uni)
    s1_xyz = np.zeros((N, 3))
    for col in range(3):
        s1_xyz[:, col] = interp1d(s1_ts_p, s1_pos[:, col],
                                   kind='linear', fill_value='extrapolate')(t_uni)
    s2_xyz = np.zeros((N, 3))
    for col in range(3):
        s2_xyz[:, col] = interp1d(s2_ts_p, s2_pos[:, col],
                                   kind='linear', fill_value='extrapolate')(t_uni)

    dist     = np.sqrt(np.sum((s1_xyz - s2_xyz) ** 2, axis=1))
    s1_pitch = interp1d(s1_ts_o,
                         np.unwrap(np.deg2rad(s1_ori[:, 1])),
                         kind='linear', fill_value='extrapolate')(t_uni)
    s1_roll  = interp1d(s1_ts_o,
                         np.unwrap(np.deg2rad(s1_ori[:, 0])),
                         kind='linear', fill_value='extrapolate')(t_uni)
    return np.column_stack([dist, s1_pitch, s1_roll,
                             s1_xyz[:, 0], s1_xyz[:, 2], s2_xyz[:, 1]])


def lowpass(sig, cut=LOWPASS_CUT, fs=FS, order=LOWPASS_ORD):
    nyq = fs / 2.0
    if cut >= nyq:
        return sig
    sos = butter(order, cut / nyq, btype='low', output='sos')
    return sosfiltfilt(sos, sig)


def load_and_build(filepath):
    s1_pos, s2_pos, s1_ori, s2_ori, s1_ts, s2_ts = parse_raw_file(filepath)
    if len(s1_pos) < 10 or len(s2_pos) < 10:
        return None
    data = build_6ch(s1_pos, s1_ori, s1_ts, s2_pos, s2_ori, s2_ts)
    if data is None:
        return None
    for col in [0, 3, 4, 5]:
        data[:, col] = lowpass(data[:, col])
    return data


def sliding_window(data, wl=WINDOW_LEN, stride=STRIDE):
    wins = []
    for s in range(0, len(data) - wl + 1, stride):
        wins.append(data[s:s + wl].copy())
    return np.array(wins) if wins else None


# ════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION (mirror preprocessing.py exactly)
# ════════════════════════════════════════════════════════════════════════════

def power_band_ratio(sig, fs, f_low, f_high):
    sig = sig - np.mean(sig)
    N   = len(sig)
    if N < 4:
        return 0.0
    Y     = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    P     = Y ** 2
    Pt    = P.sum()
    if Pt < 1e-12:
        return 0.0
    return P[(freqs >= f_low) & (freqs < f_high)].sum() / Pt


def detect_taps(sig, fs=FS):
    k        = np.ones(5) / 5.0
    smoothed = np.convolve(sig, k, mode='same')
    mu, sigma = np.mean(smoothed), np.std(smoothed)
    if sigma < 1e-8:
        return np.array([]), np.array([]), np.array([])
    peaks, _ = find_peaks(smoothed,
                           height=mu + 0.3 * sigma,
                           distance=10,
                           prominence=0.3 * sigma)
    if len(peaks) < 3:
        return np.array([]), np.array([]), np.array([])
    tap_times = peaks / fs
    return tap_times, smoothed[peaks], np.diff(tap_times)


def extract_features(win, fs=FS):
    dt       = 1.0 / fs
    duration = len(win) * dt
    dist, s1_pitch, s1_roll, s1_x, _, s2_y = (win[:, i] for i in range(6))

    tap_times, _, ITI = detect_taps(dist, fs)

    if len(ITI) > 0:
        iti_mean = np.mean(ITI)
        iti_cv   = np.std(ITI) / (iti_mean + 1e-8)
        iti_jit  = (np.mean(np.abs(np.diff(ITI))) / (iti_mean + 1e-8)
                    if len(ITI) > 1 else 0.0)
    else:
        iti_mean = iti_cv = iti_jit = 0.0
    tap_rate = len(tap_times) / duration if len(tap_times) >= 3 else 0.0

    vel = np.diff(s1_pitch) / dt

    return np.array([
        iti_mean,
        tap_rate,
        iti_cv,
        iti_jit,
        power_band_ratio(s1_roll,  fs, 2, 5),
        power_band_ratio(dist,     fs, 0, 2),
        power_band_ratio(dist,     fs, 2, 5),
        power_band_ratio(s1_x,     fs, 0, 2),
        power_band_ratio(s1_roll,  fs, 0, 2),
        power_band_ratio(s1_x,     fs, 2, 5),
        power_band_ratio(s2_y,     fs, 0, 2),
        power_band_ratio(s1_pitch, fs, 0, 2),
        np.sqrt(np.mean(vel ** 2)),
        np.std(vel),
        np.percentile(np.abs(vel), 95),
    ], dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  LOAD ALL DATA
# ════════════════════════════════════════════════════════════════════════════

def load_class(cls, max_files=None):
    folder = os.path.join(DATA_DIR, str(cls))
    files  = sorted(f for f in os.listdir(folder) if f.endswith('.txt'))
    if max_files:
        files = files[:max_files]

    signals, feature_mats, fnames = [], [], []
    for fname in files:
        fp   = os.path.join(folder, fname)
        data = load_and_build(fp)
        if data is None:
            continue
        wins = sliding_window(data)
        if wins is None:
            continue
        feats = np.array([extract_features(w) for w in wins])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        signals.append(data)
        feature_mats.append(feats)
        fnames.append(fname)
    return signals, feature_mats, fnames


print("Loading class 0 …")
sig0, feat0, fn0 = load_class(0)
print(f"  {len(sig0)} recordings loaded from class 0")

print("Loading class 1 …")
sig1, feat1, fn1 = load_class(1)
print(f"  {len(sig1)} recordings loaded from class 1")

# Flatten to windows
F0 = np.vstack(feat0)   # (N0_windows, 15)
F1 = np.vstack(feat1)   # (N1_windows, 15)
print(f"\nWindows — class 0: {len(F0)}, class 1: {len(F1)}")

# ════════════════════════════════════════════════════════════════════════════
#  FIG 1 — RAW 6-CHANNEL SIGNALS: class 0 vs class 1 (3 samples each)
# ════════════════════════════════════════════════════════════════════════════

def plot_raw_signals(sig0, sig1, fn0, fn1, n_samples=3):
    fig, axes = plt.subplots(6, n_samples * 2,
                              figsize=(22, 18),
                              sharey='row')
    fig.suptitle('Raw 6-Channel Signals\n'
                 'Left columns: UPDRS=0 (no bradykinesia)  |  '
                 'Right columns: UPDRS=1 (mild bradykinesia)',
                 fontsize=13, fontweight='bold')

    cols_0 = range(n_samples)
    cols_1 = range(n_samples, n_samples * 2)
    t_max  = 30  # seconds to display

    clr = ['tab:blue', 'tab:orange', 'tab:green',
           'tab:red', 'tab:purple', 'tab:brown']

    for col, (sig, fn) in enumerate(
            [(sig0[i % len(sig0)], fn0[i % len(fn0)]) for i in range(n_samples)] +
            [(sig1[i % len(sig1)], fn1[i % len(fn1)]) for i in range(n_samples)]):

        n_pts = min(len(sig), int(t_max * FS))
        t     = np.arange(n_pts) / FS
        label = 'UPDRS=0' if col < n_samples else 'UPDRS=1'
        bg    = '#eaf4ff' if col < n_samples else '#fff0f0'

        for ch in range(6):
            ax = axes[ch, col]
            ax.set_facecolor(bg)
            ax.plot(t, sig[:n_pts, ch], color=clr[ch], lw=0.8, alpha=0.9)
            ax.set_xlim(0, t_max)
            if col == 0:
                ax.set_ylabel(VAE_CH_NAMES[ch], fontsize=9)
            if ch == 0:
                short = os.path.basename(fn)[:28]
                ax.set_title(f'{label}\n{short}', fontsize=7.5, pad=2)
            if ch == 5:
                ax.set_xlabel('Time (s)', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, lw=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, 'fig1_raw_signals.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


plot_raw_signals(sig0, sig1, fn0, fn1)


# ════════════════════════════════════════════════════════════════════════════
#  FIG 2 — PERIODICITY ANALYSIS: FFT + autocorrelation on 'dist'
# ════════════════════════════════════════════════════════════════════════════

def dominant_period(sig_dist, fs=FS):
    """Return dominant period (seconds) from FFT of dist signal."""
    s   = sig_dist - np.mean(sig_dist)
    N   = len(s)
    Y   = np.abs(np.fft.rfft(s))
    f   = np.fft.rfftfreq(N, d=1.0 / fs)
    # Search in 0.5 – 5 Hz (physiological tapping range)
    mask = (f >= 0.5) & (f <= 5.0)
    if not mask.any():
        return np.nan
    idx = np.argmax(Y[mask])
    f_dom = f[mask][idx]
    return 1.0 / f_dom if f_dom > 0 else np.nan


def plot_periodicity(sig0, sig1, fn0, fn1, n_samples=4):
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle('Periodicity Analysis of "dist" (Finger Distance Signal)\n'
                 'UPDRS=0 (blue)  vs  UPDRS=1 (red)',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(4, n_samples * 2 + 1,
                           figure=fig, hspace=0.5, wspace=0.35)

    # ── rows 0-1: FFT spectrum ──────────────────────────────────────────────
    periods_0, periods_1 = [], []

    for col, (sig, fn, cls) in enumerate(
            [(sig0[i % len(sig0)], fn0[i % len(fn0)], 0) for i in range(n_samples)] +
            [(sig1[i % len(sig1)], fn1[i % len(fn1)], 1) for i in range(n_samples)]):

        dist = sig[:, 0]
        row_fft  = 0
        row_acor = 2
        col_idx  = col

        # — FFT —
        ax_fft = fig.add_subplot(gs[row_fft, col_idx])
        s    = dist - np.mean(dist)
        N    = len(s)
        Y    = np.abs(np.fft.rfft(s)) ** 2
        f    = np.fft.rfftfreq(N, d=1.0 / FS)
        mask = (f >= 0.2) & (f <= 6.0)
        clr  = 'tab:blue' if cls == 0 else 'tab:red'
        ax_fft.plot(f[mask], Y[mask], color=clr, lw=0.9)
        T_dom = dominant_period(dist)
        if T_dom and not np.isnan(T_dom):
            ax_fft.axvline(1.0 / T_dom, color='k', lw=1.2,
                           ls='--', alpha=0.7,
                           label=f'f={1/T_dom:.2f}Hz\nT={T_dom:.2f}s')
            ax_fft.legend(fontsize=6, loc='upper right')
            if cls == 0:
                periods_0.append(T_dom)
            else:
                periods_1.append(T_dom)
        ax_fft.set_title(f'FFT (UPDRS={cls})\n{os.path.basename(fn)[:22]}',
                          fontsize=7)
        ax_fft.set_xlabel('Freq (Hz)', fontsize=7)
        if col_idx == 0: ax_fft.set_ylabel('Power', fontsize=7)
        ax_fft.tick_params(labelsize=6)
        ax_fft.grid(True, alpha=0.3)

        # — Raw dist + tap marks —
        ax_dist = fig.add_subplot(gs[row_fft + 1, col_idx])
        t_disp  = min(len(dist), int(30 * FS))
        t       = np.arange(t_disp) / FS
        ax_dist.plot(t, dist[:t_disp], color=clr, lw=0.8, alpha=0.8)

        tap_times, tap_amps, ITI = detect_taps(dist[:t_disp], FS)
        if len(tap_times) > 0:
            ax_dist.vlines(tap_times, ymin=dist[:t_disp].min(),
                           ymax=dist[:t_disp].max(),
                           color='gray', lw=0.6, alpha=0.5)
        ax_dist.set_xlabel('Time (s)', fontsize=7)
        if col_idx == 0: ax_dist.set_ylabel('dist (cm)', fontsize=7)
        ax_dist.tick_params(labelsize=6)
        ax_dist.grid(True, alpha=0.3)

        # — Autocorrelation —
        ax_ac = fig.add_subplot(gs[row_acor, col_idx])
        seg   = dist[:int(20 * FS)]
        seg   = (seg - np.mean(seg))
        N_ac  = len(seg)
        ac    = np.correlate(seg, seg, mode='full')[N_ac - 1:]
        if ac[0] > 0:
            ac = ac / ac[0]
        lags = np.arange(len(ac)) / FS
        ax_ac.plot(lags[:int(5 * FS)], ac[:int(5 * FS)],
                   color=clr, lw=0.9)
        # mark first positive peak (periodicity)
        peaks_ac, _ = find_peaks(ac[1:int(5*FS)], height=0.05, distance=5)
        if len(peaks_ac) > 0:
            T_ac = (peaks_ac[0] + 1) / FS
            ax_ac.axvline(T_ac, color='k', ls='--', lw=1.2,
                          label=f'T={T_ac:.2f}s')
            ax_ac.legend(fontsize=6, loc='upper right')
        ax_ac.set_title(f'Autocorr (UPDRS={cls})', fontsize=7)
        ax_ac.set_xlabel('Lag (s)', fontsize=7)
        if col_idx == 0: ax_ac.set_ylabel('ACF', fontsize=7)
        ax_ac.tick_params(labelsize=6)
        ax_ac.grid(True, alpha=0.3)

    # — Summary: period distribution —
    ax_sum = fig.add_subplot(gs[3, :])
    if periods_0:
        ax_sum.hist(periods_0, bins=20, alpha=0.65, color='tab:blue',
                    label=f'UPDRS=0 (n={len(periods_0)}) '
                          f'mean={np.mean(periods_0):.2f}s')
    if periods_1:
        ax_sum.hist(periods_1, bins=20, alpha=0.65, color='tab:red',
                    label=f'UPDRS=1 (n={len(periods_1)}) '
                          f'mean={np.mean(periods_1):.2f}s')
    ax_sum.set_xlabel('Dominant Tap Period (s)', fontsize=10)
    ax_sum.set_ylabel('Count (recordings)', fontsize=10)
    ax_sum.set_title('Distribution of Dominant Tapping Period per Recording',
                     fontsize=11, fontweight='bold')
    ax_sum.legend(fontsize=9)
    ax_sum.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, 'fig2_periodicity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")
    return periods_0, periods_1


p0, p1 = plot_periodicity(sig0, sig1, fn0, fn1)


# ════════════════════════════════════════════════════════════════════════════
#  FIG 3 — FEATURE BOXPLOTS: all 15 features (per window)
# ════════════════════════════════════════════════════════════════════════════

def plot_boxplots(F0, F1):
    n_feat = len(FEAT_NAMES)
    ncols = 5
    nrows = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12))
    fig.suptitle('Feature Distributions per Window\n'
                 'Blue=UPDRS 0 (no bradykinesia)  |  Red=UPDRS 1 (mild bradykinesia)',
                 fontsize=13, fontweight='bold')

    axes_flat = axes.flatten()

    for i, name in enumerate(FEAT_NAMES):
        ax = axes_flat[i]
        v0 = F0[:, i]
        v1 = F1[:, i]

        # Remove extreme outliers for display (keep within 1-99 pctile)
        lo = min(np.percentile(v0, 1), np.percentile(v1, 1))
        hi = max(np.percentile(v0, 99), np.percentile(v1, 99))
        v0c = np.clip(v0, lo, hi)
        v1c = np.clip(v1, lo, hi)

        bp = ax.boxplot([v0c, v1c],
                         patch_artist=True,
                         widths=0.55,
                         medianprops=dict(color='black', lw=2),
                         whiskerprops=dict(lw=1.2),
                         capprops=dict(lw=1.2),
                         flierprops=dict(marker='o', ms=2, alpha=0.3))
        bp['boxes'][0].set_facecolor('#aecde8')
        bp['boxes'][1].set_facecolor('#f4a0a0')

        # Cohen's d
        m0, m1 = np.mean(v0), np.mean(v1)
        s_pool = np.sqrt((np.std(v0)**2 + np.std(v1)**2) / 2 + 1e-12)
        d      = abs(m0 - m1) / s_pool

        ax.set_title(f'{name}\n|d|={d:.3f}', fontsize=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['0', '1'], fontsize=8)
        ax.set_xlabel('UPDRS', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    # hide extra axes
    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, 'fig3_feature_boxplots.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


plot_boxplots(F0, F1)


# ════════════════════════════════════════════════════════════════════════════
#  FIG 4 — CORRELATION MATRIX of 15 features
# ════════════════════════════════════════════════════════════════════════════

def plot_correlation(F0, F1):
    Fcat  = np.vstack([F0, F1])
    corr  = np.corrcoef(Fcat.T)
    mask  = np.isnan(corr)
    corr[mask] = 0.0

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    ticks = range(len(FEAT_NAMES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(FEAT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(FEAT_NAMES, fontsize=8)

    for i in range(len(FEAT_NAMES)):
        for j in range(len(FEAT_NAMES)):
            val = corr[i, j]
            c   = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color=c)

    ax.set_title('Feature Correlation Matrix (all windows, both classes)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_correlation_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


plot_correlation(F0, F1)


# ════════════════════════════════════════════════════════════════════════════
#  FIG 5 — FEATURE HISTOGRAMS with KDE
# ════════════════════════════════════════════════════════════════════════════

def plot_histograms(F0, F1):
    from scipy.stats import gaussian_kde

    n_feat = len(FEAT_NAMES)
    ncols  = 5
    nrows  = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 13))
    fig.suptitle('Feature Histograms + KDE\n'
                 'Blue=UPDRS 0  |  Red=UPDRS 1',
                 fontsize=13, fontweight='bold')

    axes_flat = axes.flatten()
    for i, name in enumerate(FEAT_NAMES):
        ax  = axes_flat[i]
        v0  = F0[:, i]
        v1  = F1[:, i]
        lo  = min(np.percentile(v0, 1),  np.percentile(v1, 1))
        hi  = max(np.percentile(v0, 99), np.percentile(v1, 99))
        v0c = np.clip(v0, lo, hi)
        v1c = np.clip(v1, lo, hi)

        bins = np.linspace(lo, hi, 30)
        ax.hist(v0c, bins=bins, density=True, alpha=0.45,
                color='tab:blue', label='UPDRS 0')
        ax.hist(v1c, bins=bins, density=True, alpha=0.45,
                color='tab:red',  label='UPDRS 1')

        try:
            if np.std(v0c) > 1e-8:
                kde0 = gaussian_kde(v0c)
                xs   = np.linspace(lo, hi, 200)
                ax.plot(xs, kde0(xs), 'tab:blue', lw=1.8)
            if np.std(v1c) > 1e-8:
                kde1 = gaussian_kde(v1c)
                xs   = np.linspace(lo, hi, 200)
                ax.plot(xs, kde1(xs), 'tab:red',  lw=1.8)
        except Exception:
            pass

        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, 'fig5_feature_histograms.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


plot_histograms(F0, F1)


# ════════════════════════════════════════════════════════════════════════════
#  FIG 6 — MEAN ± STD bar chart for all 15 features
# ════════════════════════════════════════════════════════════════════════════

def plot_mean_std(F0, F1):
    # z-score for display comparability
    Fcat = np.vstack([F0, F1])
    mu   = Fcat.mean(axis=0)
    sig  = Fcat.std(axis=0)
    sig[sig < 1e-8] = 1.0
    F0z  = (F0 - mu) / sig
    F1z  = (F1 - mu) / sig

    m0   = F0z.mean(axis=0)
    m1   = F1z.mean(axis=0)
    s0   = F0z.std(axis=0)
    s1   = F1z.std(axis=0)

    x    = np.arange(len(FEAT_NAMES))
    w    = 0.38

    fig, ax = plt.subplots(figsize=(18, 7))
    b0 = ax.bar(x - w / 2, m0, w, yerr=s0 / np.sqrt(len(F0)),
                capsize=3, color='#6baed6', alpha=0.85,
                label='UPDRS=0', error_kw=dict(elinewidth=1))
    b1 = ax.bar(x + w / 2, m1, w, yerr=s1 / np.sqrt(len(F1)),
                capsize=3, color='#fc8d59', alpha=0.85,
                label='UPDRS=1', error_kw=dict(elinewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels(FEAT_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Z-scored feature value (mean ± SEM)', fontsize=10)
    ax.set_title('Mean ± SEM of all 15 Features (z-scored)\n'
                 'Blue=UPDRS 0  |  Orange=UPDRS 1',
                 fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', lw=0.7, ls='--')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_mean_std_bar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


plot_mean_std(F0, F1)


# ════════════════════════════════════════════════════════════════════════════
#  FIG 7 — DOMINANT TAP PERIOD: all recordings (FFT-based)
# ════════════════════════════════════════════════════════════════════════════

def compute_all_periods(sigs, cls):
    rows = []
    for i, sig in enumerate(sigs):
        T = dominant_period(sig[:, 0])
        if T and not np.isnan(T) and 0.1 < T < 3.0:
            rows.append(T)
    return np.array(rows)


def plot_period_distribution(sig0, sig1):
    T0 = compute_all_periods(sig0, 0)
    T1 = compute_all_periods(sig1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Dominant Tapping Period per Recording (FFT on dist channel)',
                 fontsize=13, fontweight='bold')

    for ax, T, label, clr in [
            (axes[0], T0, 'UPDRS=0', 'tab:blue'),
            (axes[1], T1, 'UPDRS=1', 'tab:red')]:
        if len(T) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            continue
        bins = np.arange(0.1, 3.0, 0.1)
        ax.hist(T, bins=bins, color=clr, alpha=0.75, edgecolor='white', lw=0.5)
        ax.axvline(np.mean(T),   color='black', lw=2,   ls='--',
                   label=f'mean={np.mean(T):.2f}s')
        ax.axvline(np.median(T), color='gold',  lw=1.5, ls=':',
                   label=f'median={np.median(T):.2f}s')
        ax.set_xlabel('Dominant Tap Period (s)', fontsize=11)
        ax.set_ylabel('Number of recordings', fontsize=11)
        ax.set_title(f'{label}  (n={len(T)} recordings)\n'
                     f'mean={np.mean(T):.3f}s   '
                     f'median={np.median(T):.3f}s   '
                     f'std={np.std(T):.3f}s   \n'
                     f'Range: {np.min(T):.2f}–{np.max(T):.2f}s',
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig7_period_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")

    return T0, T1


T0_all, T1_all = plot_period_distribution(sig0, sig1)


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINT
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("SUMMARY STATISTICS")
print("=" * 65)
print(f"{'Feature':<28}  {'UPDRS=0':>12}  {'UPDRS=1':>12}  {'|d|':>7}")
print("-" * 65)
for i, name in enumerate(FEAT_NAMES):
    v0, v1 = F0[:, i], F1[:, i]
    m0, m1 = np.mean(v0), np.mean(v1)
    s_pool = np.sqrt((np.std(v0)**2 + np.std(v1)**2) / 2 + 1e-12)
    d = abs(m0 - m1) / s_pool
    print(f"{name:<28}  {m0:>12.4f}  {m1:>12.4f}  {d:>7.3f}")

print("\n--- Dominant Tapping Period (from FFT on dist) ---")
if len(T0_all) > 0:
    print(f"  UPDRS=0:  mean={np.mean(T0_all):.3f}s  "
          f"median={np.median(T0_all):.3f}s  "
          f"std={np.std(T0_all):.3f}s")
if len(T1_all) > 0:
    print(f"  UPDRS=1:  mean={np.mean(T1_all):.3f}s  "
          f"median={np.median(T1_all):.3f}s  "
          f"std={np.std(T1_all):.3f}s")
print("=" * 65)
print(f"\nAll figures saved to: {OUT_DIR}/")
