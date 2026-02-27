"""
visualize_features.py
=====================
Load raw files from folder 0 (UPDRS=0, Non-PD) and folder 1 (UPDRS=1, PD),
compute all 15 clinical features on the full recording (not windowed), and
produce comprehensive visualisations:

  Figure 1 – Raw signals + tap detection  (1 file each class)
  Figure 2 – FFT spectra + dominant-frequency heatmap
  Figure 3 – Autocorrelation analysis (periodicity)
  Figure 4 – ITI (inter-tap-interval) distributions → tapping period
  Figure 5 – 15 feature distributions: class 0 vs class 1 violin plots
  Figure 6 – Feature correlation matrix for each class
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.signal import correlate

# ── Reuse the project's preprocessing pipeline ─────────────────────────────
sys.path.insert(0, '/home/user/Final-work')
from preprocessing import (
    parse_raw_file, resample_and_build_6ch, lowpass_filter,
    detect_taps_on_signal, power_band_ratio,
    extract_window_features, Config
)

FS   = Config.FS          # 60 Hz
DT   = 1.0 / FS
OUT  = '/home/user/Final-work/results/feature_vis'
os.makedirs(OUT, exist_ok=True)

FOLDER = {0: '/home/user/Final-work/0', 1: '/home/user/Final-work/1'}
COLORS = {0: '#2196F3', 1: '#E53935'}   # blue / red
LABELS = {0: 'Non-PD (UPDRS=0)', 1: 'PD (UPDRS=1)'}

FEAT_NAMES = Config.CLINICAL_FEATURE_NAMES   # 15 names
CH_NAMES   = Config.VAE_CHANNEL_NAMES        # 6 channel names


# ========================== DATA LOADING ===================================

def load_full_recording(filepath):
    """Parse + resample one file → (N, 6) full-length signal, no windowing."""
    s1_pos, s2_pos, s1_ori, s2_ori, s1_ts, s2_ts = parse_raw_file(filepath)
    if len(s1_pos) < 20:
        return None
    data = resample_and_build_6ch(s1_pos, s1_ori, s1_ts,
                                   s2_pos, s2_ori, s2_ts, FS)
    if data is None or len(data) < FS * 3:
        return None
    for col in [0, 3, 4, 5]:   # lowpass position channels
        data[:, col] = lowpass_filter(data[:, col], Config.LOWPASS_CUTOFF,
                                       FS, Config.LOWPASS_ORDER)
    return data


def load_all_files(cls, max_files=None):
    folder = FOLDER[cls]
    files  = sorted(f for f in os.listdir(folder) if f.endswith('.txt'))
    if max_files:
        files = files[:max_files]
    recordings = []
    for fname in files:
        data = load_full_recording(os.path.join(folder, fname))
        if data is not None:
            recordings.append((fname, data))
    print(f"  Class {cls}: loaded {len(recordings)}/{len(files)} files")
    return recordings


# ========================== FEATURE EXTRACTION (WHOLE FILE) ================

def extract_full_file_features(data):
    """Extract features on the whole recording as one big window."""
    return extract_window_features(data, fs=FS)


def collect_all_features(recordings):
    """(N_files, 15) feature matrix for one class."""
    feat_list = []
    for fname, data in recordings:
        f = extract_full_file_features(data)
        feat_list.append(f)
    return np.array(feat_list)   # (N, 15)


# ========================== PERIODICITY HELPERS ============================

def dominant_freq_fft(signal, fs=FS, f_min=0.5, f_max=8.0):
    """Return dominant frequency (Hz) in [f_min, f_max]."""
    sig = signal - np.mean(signal)
    N   = len(sig)
    if N < 8:
        return np.nan
    Y     = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mask  = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        return np.nan
    return freqs[mask][np.argmax(Y[mask])]


def autocorr_period(signal, fs=FS, max_lag_sec=3.0):
    """
    Estimate dominant period via autocorrelation.
    Returns period in seconds (nan if no clear peak).
    """
    sig  = signal - np.mean(signal)
    ac   = correlate(sig, sig, mode='full')
    ac   = ac[len(ac) // 2:]          # keep positive lags
    ac  /= ac[0] + 1e-12              # normalise
    max_lag = int(max_lag_sec * fs)
    ac_search = ac[5:max_lag]          # skip zero-lag
    if len(ac_search) < 5:
        return np.nan
    peaks, _ = find_peaks(ac_search, height=0.2, distance=int(0.1 * fs))
    if len(peaks) == 0:
        return np.nan
    return peaks[0] / fs              # lag of first peak → period


# ========================== FIGURE 1: Raw signals + tap detection ===========

def fig1_raw_signals(recs_0, recs_1):
    """Plot first file of each class: all 6 channels + tap peaks on dist."""
    fig, axes = plt.subplots(6, 2, figsize=(18, 18),
                              gridspec_kw={'hspace': 0.5})
    fig.suptitle('Figure 1  —  Raw Signals + Tap Detection\n'
                 '(first recording of each class)', fontsize=14, fontweight='bold')

    for col_idx, (cls, recs) in enumerate([(0, recs_0), (1, recs_1)]):
        fname, data = recs[0]
        T    = len(data)
        time = np.arange(T) * DT
        c    = COLORS[cls]

        for ch in range(6):
            ax  = axes[ch, col_idx]
            sig = data[:, ch]
            ax.plot(time, sig, color=c, lw=0.6, alpha=0.85)
            ax.set_ylabel(CH_NAMES[ch], fontsize=8)
            ax.set_xlim(time[0], time[-1])

            # Overlay tap peaks on dist channel
            if ch == 0:
                tap_times, _, ITI = detect_taps_on_signal(sig, FS)
                if len(tap_times) > 0:
                    tap_idx = (tap_times * FS).astype(int)
                    tap_idx = tap_idx[tap_idx < T]
                    ax.scatter(time[tap_idx], sig[tap_idx],
                               color='k', s=25, zorder=5, label='tap peaks')
                    mean_period = np.mean(ITI) if len(ITI) > 0 else np.nan
                    rate = len(tap_times) / (T * DT)
                    ax.set_title(
                        f'{LABELS[cls]}  |  {os.path.basename(fname)[:40]}\n'
                        f'dist: {len(tap_times)} taps, rate={rate:.2f} Hz, '
                        f'mean ITI={mean_period:.3f} s',
                        fontsize=7)
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.tick_params(labelsize=7)

    path = os.path.join(OUT, 'fig1_raw_signals.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ========================== FIGURE 2: FFT Spectra ==========================

def fig2_fft_spectra(recs_0, recs_1):
    """FFT of dist and s1_pitch for every file; heatmap + mean spectra."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Figure 2  —  FFT Power Spectra\n'
                 '(dist channel  |  mean across all files per class)',
                 fontsize=13, fontweight='bold')

    f_plot_max = 10.0   # Hz

    for row, (ch_idx, ch_name) in enumerate([(0, 'dist'), (1, 's1_pitch')]):
        for col_idx, (cls, recs) in enumerate([(0, recs_0), (1, recs_1)]):
            ax = axes[row, col_idx]
            all_psd = []
            freqs_ref = None

            for _, data in recs:
                sig   = data[:, ch_idx] - np.mean(data[:, ch_idx])
                N     = len(sig)
                Y     = np.abs(np.fft.rfft(sig)) ** 2
                freqs = np.fft.rfftfreq(N, d=DT)
                # Interpolate to common frequency grid
                if freqs_ref is None:
                    freqs_ref = freqs
                if len(freqs) == len(freqs_ref):
                    all_psd.append(Y)

            if all_psd:
                psd_mat = np.array(all_psd)
                # Normalise each row to sum=1
                row_sums = psd_mat.sum(axis=1, keepdims=True)
                psd_norm = psd_mat / (row_sums + 1e-12)

                mask = freqs_ref <= f_plot_max
                mean_psd = psd_norm.mean(axis=0)
                std_psd  = psd_norm.std(axis=0)

                ax.fill_between(freqs_ref[mask],
                                (mean_psd - std_psd)[mask],
                                (mean_psd + std_psd)[mask],
                                alpha=0.25, color=COLORS[cls])
                ax.plot(freqs_ref[mask], mean_psd[mask],
                        color=COLORS[cls], lw=1.8,
                        label=f'mean ± std  (n={len(all_psd)})')

                # Mark dominant frequency
                search = (freqs_ref >= 0.5) & (freqs_ref <= 8.0) & (freqs_ref <= f_plot_max)
                if search.any():
                    dom_f = freqs_ref[search][np.argmax(mean_psd[search])]
                    ax.axvline(dom_f, color='k', ls='--', lw=1)
                    ax.text(dom_f + 0.1, ax.get_ylim()[1] * 0.95,
                            f'{dom_f:.2f} Hz\n({1/dom_f:.2f} s)',
                            fontsize=8, va='top')

                # Shade clinical bands
                ax.axvspan(0, 2,   alpha=0.07, color='blue',  label='0–2 Hz band')
                ax.axvspan(2, 5,   alpha=0.07, color='green', label='2–5 Hz band')

            ax.set_xlabel('Frequency (Hz)', fontsize=9)
            ax.set_ylabel('Normalised Power', fontsize=9)
            ax.set_title(f'{ch_name}  |  {LABELS[cls]}', fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlim(0, f_plot_max)

    path = os.path.join(OUT, 'fig2_fft_spectra.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ========================== FIGURE 3: Autocorrelation =====================

def fig3_autocorr(recs_0, recs_1):
    """Autocorrelation of dist for a few example files + mean."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Figure 3  —  Autocorrelation of Finger Distance (dist)\n'
                 '(periodicity analysis — first peak lag = tapping period)',
                 fontsize=13, fontweight='bold')

    max_lag_sec = 3.0
    max_lag_idx = int(max_lag_sec * FS)

    for col_idx, (cls, recs) in enumerate([(0, recs_0), (1, recs_1)]):
        ax      = axes[col_idx]
        periods = []
        all_ac  = []

        for k, (fname, data) in enumerate(recs):
            sig = data[:, 0] - np.mean(data[:, 0])
            ac  = correlate(sig, sig, mode='full')
            half = len(ac) // 2
            ac  = ac[half:half + max_lag_idx + 1]
            if len(ac) == 0:
                continue
            ac /= ac[0] + 1e-12

            if k < 8:   # plot first 8 individuals lightly
                lag_t = np.arange(len(ac)) * DT
                ax.plot(lag_t, ac, color=COLORS[cls], lw=0.6, alpha=0.3)

            all_ac.append(ac)

            period = autocorr_period(data[:, 0])
            if not np.isnan(period):
                periods.append(period)

        # Mean autocorrelation
        min_len = min(len(a) for a in all_ac)
        mean_ac = np.mean([a[:min_len] for a in all_ac], axis=0)
        lag_t   = np.arange(min_len) * DT
        ax.plot(lag_t, mean_ac, color=COLORS[cls], lw=2.5, label='Mean AC')
        ax.axhline(0, color='k', lw=0.8, ls='--')

        # Mark first peak of mean AC
        search_ac = mean_ac[5:int(2.5 * FS)]
        peaks, _  = find_peaks(search_ac, height=0.1, distance=int(0.1 * FS))
        if len(peaks) > 0:
            p_lag = (peaks[0] + 5) * DT
            ax.axvline(p_lag, color='red', lw=1.5, ls='-.',
                       label=f'1st peak = {p_lag:.3f} s')
            ax.scatter([p_lag], [mean_ac[peaks[0] + 5]], color='red', s=80, zorder=5)

        med_period = np.median(periods) if periods else np.nan
        ax.set_title(f'{LABELS[cls]}\n'
                     f'Median tapping period = {med_period:.3f} s  '
                     f'({1/med_period:.2f} Hz)' if not np.isnan(med_period)
                     else f'{LABELS[cls]}', fontsize=10)
        ax.set_xlabel('Lag (s)', fontsize=9)
        ax.set_ylabel('Autocorrelation', fontsize=9)
        ax.set_xlim(0, max_lag_sec)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.6, 1.05)

    path = os.path.join(OUT, 'fig3_autocorr.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ========================== FIGURE 4: ITI Distributions ====================

def fig4_iti(recs_0, recs_1):
    """ITI distributions — reveals tapping period directly."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 4  —  Inter-Tap Interval (ITI) Analysis\n'
                 'ITI = time between consecutive finger-open peaks → directly = tapping period',
                 fontsize=13, fontweight='bold')

    all_iti = {0: [], 1: []}
    dom_freqs = {0: [], 1: []}

    for cls, recs in [(0, recs_0), (1, recs_1)]:
        for fname, data in recs:
            _, _, ITI = detect_taps_on_signal(data[:, 0], FS)
            all_iti[cls].extend(ITI.tolist())
            df = dominant_freq_fft(data[:, 0])
            if not np.isnan(df):
                dom_freqs[cls].append(df)

    # ── Panel A: ITI histogram both classes ─────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 2.0, 60)
    for cls in [0, 1]:
        iti = np.array(all_iti[cls])
        iti = iti[(iti > 0.05) & (iti < 2.0)]
        med = np.median(iti) if len(iti) > 0 else np.nan
        ax.hist(iti, bins=bins, alpha=0.55, color=COLORS[cls],
                label=f'{LABELS[cls]}\nmedian={med:.3f} s ({1/med:.2f} Hz)',
                density=True)
    ax.set_xlabel('ITI (s)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('ITI Distribution (= tapping period)', fontsize=10)
    ax.legend(fontsize=8)
    ax.axvline(0.5, color='gray', ls=':', lw=1, label='0.5 s = 2 Hz')

    # ── Panel B: dominant FFT frequency ─────────────────────────────────
    ax = axes[1]
    for cls in [0, 1]:
        df_arr = np.array(dom_freqs[cls])
        med_f  = np.median(df_arr)
        ax.hist(df_arr, bins=30, alpha=0.55, color=COLORS[cls],
                label=f'{LABELS[cls]}\nmedian={med_f:.2f} Hz ({1/med_f:.2f} s)',
                density=True)
    ax.set_xlabel('Dominant Frequency in dist (Hz)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Dominant FFT Frequency (0.5–8 Hz)', fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel C: ITI over time (first file each class) ──────────────────
    ax = axes[2]
    for cls, recs in [(0, recs_0), (1, recs_1)]:
        _, data = recs[0]
        tap_times, _, ITI = detect_taps_on_signal(data[:, 0], FS)
        if len(ITI) > 0:
            mid_times = (tap_times[:-1] + tap_times[1:]) / 2
            ax.plot(mid_times, ITI, 'o-', color=COLORS[cls],
                    ms=4, lw=1.2, alpha=0.8, label=LABELS[cls])
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('ITI (s)', fontsize=10)
    ax.set_title('ITI Over Time\n(fatigue / freezing: rising ITI)', fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(1.0, color='gray', ls=':', lw=1)

    path = os.path.join(OUT, 'fig4_iti.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ========================== FIGURE 5: 15 Feature Violin Plots ==============

def fig5_feature_violins(feat0, feat1):
    """Violin + strip plot for all 15 features, class 0 vs class 1."""
    n_feat = len(FEAT_NAMES)
    ncols  = 5
    nrows  = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 4))
    fig.suptitle('Figure 5  —  15 Clinical Feature Distributions\n'
                 'Blue = Non-PD (UPDRS=0)   |   Red = PD (UPDRS=1)',
                 fontsize=13, fontweight='bold')
    axes = axes.flat

    for i, fname in enumerate(FEAT_NAMES):
        ax   = axes[i]
        v0   = feat0[:, i]
        v1   = feat1[:, i]

        # Cohen's d
        pooled_std = np.sqrt((np.std(v0) ** 2 + np.std(v1) ** 2) / 2 + 1e-12)
        cohen_d    = abs(np.mean(v0) - np.mean(v1)) / pooled_std

        parts = ax.violinplot([v0, v1], positions=[0, 1],
                               showmedians=True, showextrema=True)
        parts['bodies'][0].set_facecolor(COLORS[0])
        parts['bodies'][1].set_facecolor(COLORS[1])
        for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if key in parts:
                parts[key].set_color('k')
                parts[key].set_linewidth(1.2)

        # Jitter strip plot
        rng = np.random.default_rng(42 + i)
        for pos, vals, c in [(0, v0, COLORS[0]), (1, v1, COLORS[1])]:
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(pos + jitter, vals, color=c, alpha=0.3, s=6, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Non-PD', 'PD'], fontsize=8)
        ax.set_title(f'[{i+1}] {fname}\n|d|={cohen_d:.3f}', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

    for j in range(i + 1, len(list(axes))):
        axes[j].axis('off')

    path = os.path.join(OUT, 'fig5_feature_violins.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ========================== FIGURE 6: Periodicity Summary ==================

def fig6_period_summary(recs_0, recs_1):
    """
    Per-file dominant period from three methods:
      (a) 1/mean(ITI)   (b) 1/FFT_dom_freq   (c) autocorr lag
    Scatter and box plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Figure 6  —  Tapping Period Estimation (3 Methods)\n'
                 'All methods should agree if signal is truly periodic',
                 fontsize=13, fontweight='bold')

    method_data = {0: {'iti': [], 'fft': [], 'ac': []},
                   1: {'iti': [], 'fft': [], 'ac': []}}

    for cls, recs in [(0, recs_0), (1, recs_1)]:
        for fname, data in recs:
            dist = data[:, 0]
            _, _, ITI = detect_taps_on_signal(dist, FS)
            if len(ITI) >= 2:
                method_data[cls]['iti'].append(np.mean(ITI))
            fft_f = dominant_freq_fft(dist)
            if not np.isnan(fft_f):
                method_data[cls]['fft'].append(1.0 / fft_f)
            ac_p = autocorr_period(dist)
            if not np.isnan(ac_p):
                method_data[cls]['ac'].append(ac_p)

    method_keys   = ['iti', 'fft', 'ac']
    method_labels = ['Mean ITI (direct)', 'FFT dominant (1/f)', 'Autocorr lag']

    for ax, key, lbl in zip(axes, method_keys, method_labels):
        data_to_plot = [method_data[0][key], method_data[1][key]]
        bps = ax.boxplot(data_to_plot, patch_artist=True,
                         medianprops={'color': 'k', 'linewidth': 2})
        for patch, cls in zip(bps['boxes'], [0, 1]):
            patch.set_facecolor(COLORS[cls])
            patch.set_alpha(0.6)
        for cls, vals in enumerate(data_to_plot):
            rng = np.random.default_rng(77 + cls)
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(cls + 1 + jitter, vals,
                       color=COLORS[cls], alpha=0.5, s=20, zorder=3)

        # Print medians
        for cls in [0, 1]:
            vals = method_data[cls][key]
            if vals:
                med = np.median(vals)
                ax.text(cls + 1, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.02,
                        f'med={med:.3f}s\n({1/med:.2f}Hz)',
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(fc='white', alpha=0.7, pad=2))

        ax.set_xticks([1, 2])
        ax.set_xticklabels([LABELS[0], LABELS[1]], fontsize=8)
        ax.set_ylabel('Period (s)', fontsize=10)
        ax.set_title(lbl, fontsize=10)
        ax.set_ylim(bottom=0)

    path = os.path.join(OUT, 'fig6_period_summary.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ========================== PRINT SUMMARY ==================================

def print_periodicity_summary(recs_0, recs_1):
    print(f"\n{'='*65}")
    print(f"  PERIODICITY SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Method':<28s} {'Non-PD (cls0)':>18s} {'PD (cls1)':>15s}")
    print(f"  {'-'*63}")

    for cls, recs in [(0, recs_0), (1, recs_1)]:
        itimeans, fftfreqs, acperiods = [], [], []
        for fname, data in recs:
            dist = data[:, 0]
            _, _, ITI = detect_taps_on_signal(dist, FS)
            if len(ITI) >= 2:
                itimeans.append(np.mean(ITI))
            f = dominant_freq_fft(dist)
            if not np.isnan(f):
                fftfreqs.append(f)
            p = autocorr_period(dist)
            if not np.isnan(p):
                acperiods.append(p)

        label = LABELS[cls]
        print(f"\n  {label}")
        print(f"  {'ITI mean period':28s} {np.median(itimeans):.3f} s   "
              f"({1/np.median(itimeans):.2f} Hz)")
        print(f"  {'FFT dominant period':28s} {np.median(1/np.array(fftfreqs)):.3f} s   "
              f"({np.median(fftfreqs):.2f} Hz)")
        print(f"  {'Autocorr period':28s} {np.median(acperiods):.3f} s   "
              f"({1/np.median(acperiods):.2f} Hz)")


# ========================== MAIN ===========================================

def main():
    print("Loading recordings...")
    recs_0 = load_all_files(0)
    recs_1 = load_all_files(1)
    print(f"  Class 0: {len(recs_0)} files  |  Class 1: {len(recs_1)} files")

    print("\nExtracting full-recording features...")
    feat0 = collect_all_features(recs_0)
    feat1 = collect_all_features(recs_1)
    print(f"  feat0={feat0.shape}  feat1={feat1.shape}")

    print("\nGenerating figures...")
    print("  Figure 1: raw signals + tap detection")
    fig1_raw_signals(recs_0, recs_1)

    print("  Figure 2: FFT spectra")
    fig2_fft_spectra(recs_0, recs_1)

    print("  Figure 3: autocorrelation")
    fig3_autocorr(recs_0, recs_1)

    print("  Figure 4: ITI distributions")
    fig4_iti(recs_0, recs_1)

    print("  Figure 5: feature violin plots")
    fig5_feature_violins(feat0, feat1)

    print("  Figure 6: period estimation summary")
    fig6_period_summary(recs_0, recs_1)

    print_periodicity_summary(recs_0, recs_1)

    print(f"\n{'='*65}")
    print(f"  All figures saved to: {OUT}/")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
