"""
data_quality_check.py
=====================
Diagnose every recording in data/0/ and data/1/ for problems
that will corrupt machine-learning downstream:

  P1  Tap-detection failure  — dist_tap_rate = 0 for ALL windows
  P2  Near-zero dist range   — finger barely moves (< 1 cm peak-to-peak)
  P3  Flat / constant signal — any channel has std < threshold
  P4  Short recording        — < 2 windows (< 2 × 5 s)
  P5  Extreme feature outlier— any feature value > 6 sigma from class mean
  P6  NaN / Inf in features  — preprocessing produces invalid numbers
  P7  Implausible tap rate   — rate < 0.3 Hz or > 6 Hz (physically impossible)
  P8  Timestamp anomaly      — duplicate / monotonicity break > 10 % of rows

Output
------
  results/quality/quality_report.txt   full per-file report
  results/quality/fig_quality_*.png    diagnostic plots
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

DATA_DIR = "data"
OUT_DIR  = "results/quality"
os.makedirs(OUT_DIR, exist_ok=True)

FS          = 60
LOWPASS_CUT = 20.0
LOWPASS_ORD = 4
WINDOW_LEN  = int(5.0 * FS)   # 300
STRIDE      = int(WINDOW_LEN * (1 - 0.75))  # 75

FEAT_NAMES = [
    'dist_iti_mean','dist_tap_rate','dist_iti_cv','dist_tap_jitter',
    's1_roll_pow_2_5hz','dist_pow_0_2hz','dist_pow_2_5hz',
    's1_x_pow_0_2hz','s1_roll_pow_0_2hz','s1_x_pow_2_5hz',
    's2_y_pow_0_2hz','s1_pitch_pow_0_2hz',
    's1_pitch_vel_rms','s1_pitch_vel_std','s1_pitch_vel_p95',
]

# ── low-level helpers (same as preprocessing.py) ─────────────────────────

def parse_raw(filepath):
    s1p,s2p,s1o,s2o,s1t,s2t = [],[],[],[],[],[]
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8: continue
            try:
                sid = parts[0]
                pos = [float(parts[1]),float(parts[2]),float(parts[3])]
                ori = [float(parts[4]),float(parts[5]),float(parts[6])]
                ts  = float(parts[7])
                if sid == '01':
                    s1p.append(pos); s1o.append(ori); s1t.append(ts)
                elif sid == '02':
                    s2p.append(pos); s2o.append(ori); s2t.append(ts)
            except ValueError:
                continue
    return (np.array(s1p),np.array(s2p),
            np.array(s1o),np.array(s2o),
            np.array(s1t),np.array(s2t))

def dedup(data, ts):
    ts_u, inv = np.unique(ts, return_inverse=True)
    if len(ts_u)==len(ts): return data, ts
    d2 = np.zeros((len(ts_u), data.shape[1]))
    c  = np.zeros(len(ts_u))
    for i,idx in enumerate(inv):
        d2[idx]+=data[i]; c[idx]+=1
    return d2/c[:,None], ts_u

def build_6ch(s1p,s1o,s1t,s2p,s2o,s2t):
    s1p,s1t_p = dedup(s1p,s1t)
    s1o,s1t_o = dedup(s1o,s1t)
    s2p,s2t_p = dedup(s2p,s2t)
    t0 = max(s1t_p[0],s2t_p[0]); t1 = min(s1t_p[-1],s2t_p[-1])
    if t1<=t0: return None
    tu = np.arange(t0,t1,1000./FS)
    if len(tu)<10: return None
    N = len(tu)
    s1xyz = np.zeros((N,3))
    for c in range(3):
        s1xyz[:,c]=interp1d(s1t_p,s1p[:,c],kind='linear',
                             fill_value='extrapolate')(tu)
    s2xyz = np.zeros((N,3))
    for c in range(3):
        s2xyz[:,c]=interp1d(s2t_p,s2p[:,c],kind='linear',
                             fill_value='extrapolate')(tu)
    dist    = np.sqrt(np.sum((s1xyz-s2xyz)**2,axis=1))
    s1pitch = interp1d(s1t_o,np.unwrap(np.deg2rad(s1o[:,1])),
                        kind='linear',fill_value='extrapolate')(tu)
    s1roll  = interp1d(s1t_o,np.unwrap(np.deg2rad(s1o[:,0])),
                        kind='linear',fill_value='extrapolate')(tu)
    return np.column_stack([dist,s1pitch,s1roll,
                             s1xyz[:,0],s1xyz[:,2],s2xyz[:,1]])

def lowpass(sig):
    nyq = FS/2.
    if LOWPASS_CUT>=nyq: return sig
    sos = butter(LOWPASS_ORD,LOWPASS_CUT/nyq,btype='low',output='sos')
    return sosfiltfilt(sos,sig)

def sliding_window(data):
    wins=[]
    for s in range(0,len(data)-WINDOW_LEN+1,STRIDE):
        wins.append(data[s:s+WINDOW_LEN].copy())
    return np.array(wins) if wins else None

def power_band_ratio(sig,fs,fl,fh):
    sig=sig-np.mean(sig); N=len(sig)
    if N<4: return 0.
    Y=np.abs(np.fft.rfft(sig)); P=Y**2
    f=np.fft.rfftfreq(N,d=1./fs); Pt=P.sum()
    if Pt<1e-12: return 0.
    return P[(f>=fl)&(f<fh)].sum()/Pt

def detect_taps(sig,fs=FS):
    k=np.ones(5)/5.; sm=np.convolve(sig,k,mode='same')
    mu,sd=np.mean(sm),np.std(sm)
    if sd<1e-8: return np.array([]),np.array([]),np.array([])
    peaks,_=find_peaks(sm,height=mu+.3*sd,distance=10,prominence=.3*sd)
    if len(peaks)<3: return np.array([]),np.array([]),np.array([])
    tt=peaks/fs
    return tt,sm[peaks],np.diff(tt)

def extract_features(win,fs=FS):
    dt=1./fs; dur=len(win)*dt
    d,sp,sr,sx,_,s2y = (win[:,i] for i in range(6))
    tt,_,ITI=detect_taps(d,fs)
    if len(ITI)>0:
        im=np.mean(ITI); ic=np.std(ITI)/(im+1e-8)
        ij=(np.mean(np.abs(np.diff(ITI)))/(im+1e-8) if len(ITI)>1 else 0.)
    else:
        im=ic=ij=0.
    rate=len(tt)/dur if len(tt)>=3 else 0.
    vel=np.diff(sp)/dt
    return np.array([
        im, rate, ic, ij,
        power_band_ratio(sr,fs,2,5), power_band_ratio(d,fs,0,2),
        power_band_ratio(d,fs,2,5),  power_band_ratio(sx,fs,0,2),
        power_band_ratio(sr,fs,0,2), power_band_ratio(sx,fs,2,5),
        power_band_ratio(s2y,fs,0,2),power_band_ratio(sp,fs,0,2),
        np.sqrt(np.mean(vel**2)), np.std(vel),
        np.percentile(np.abs(vel),95),
    ],dtype=np.float32)

# ── timestamp diagnostics ─────────────────────────────────────────────────

def ts_anomaly_rate(ts):
    """Fraction of samples where timestamp goes backwards or stays same."""
    if len(ts)<2: return 0.
    diff=np.diff(ts)
    return (diff<=0).sum()/(len(ts)-1)

# ════════════════════════════════════════════════════════════════════════════
#  PROCESS ALL FILES
# ════════════════════════════════════════════════════════════════════════════

problems_by_file = {}   # fname -> list of problem strings
all_records      = []   # dicts for stats

CH_NAMES = ['dist','s1_pitch','s1_roll','s1_x','s1_z','s2_y']

def check_file(cls, fname):
    fp = os.path.join(DATA_DIR, str(cls), fname)
    rec = dict(cls=cls, fname=fname, problems=[], n_windows=0,
               n_s1=0, n_s2=0, duration_s=0.,
               dist_range=0., flat_channels=[],
               tap_rate_mean=0., n_windows_zero_tap=0,
               ts_anom_s1=0., ts_anom_s2=0.,
               feat_mat=None)

    # ── parse ───────────────────────────────────────────────────────────
    s1p,s2p,s1o,s2o,s1t,s2t = parse_raw(fp)
    rec['n_s1'], rec['n_s2'] = len(s1p), len(s2p)

    if len(s1p)<10 or len(s2p)<10:
        rec['problems'].append(f'P0-too-few-samples (s1={len(s1p)}, s2={len(s2p)})')
        return rec

    # ── timestamp anomaly ────────────────────────────────────────────────
    rec['ts_anom_s1'] = ts_anomaly_rate(s1t)
    rec['ts_anom_s2'] = ts_anomaly_rate(s2t)
    if rec['ts_anom_s1']>0.10:
        rec['problems'].append(f'P8-ts-anomaly-s1 ({rec["ts_anom_s1"]:.1%} non-monotone)')
    if rec['ts_anom_s2']>0.10:
        rec['problems'].append(f'P8-ts-anomaly-s2 ({rec["ts_anom_s2"]:.1%} non-monotone)')

    # ── build 6-ch ───────────────────────────────────────────────────────
    data = build_6ch(s1p,s1o,s1t,s2p,s2o,s2t)
    if data is None:
        rec['problems'].append('P0-build-failed (no common time range)')
        return rec

    rec['duration_s'] = len(data)/FS

    # ── lowpass on position channels ─────────────────────────────────────
    for col in [0,3,4,5]:
        data[:,col] = lowpass(data[:,col])

    # ── P2: near-zero dist ───────────────────────────────────────────────
    rec['dist_range'] = float(np.ptp(data[:,0]))
    if rec['dist_range'] < 1.0:   # < 1 cm peak-to-peak
        rec['problems'].append(f'P2-tiny-dist-range ({rec["dist_range"]:.3f} cm)')

    # ── P3: flat channels ────────────────────────────────────────────────
    flat = []
    for ci, cn in enumerate(CH_NAMES):
        if np.std(data[:,ci]) < 0.02:
            flat.append(cn)
    rec['flat_channels'] = flat
    if flat:
        rec['problems'].append(f'P3-flat-channel {flat}')

    # ── windowing ────────────────────────────────────────────────────────
    wins = sliding_window(data)
    if wins is None or len(wins)==0:
        rec['problems'].append('P4-too-short (0 windows)')
        return rec

    rec['n_windows'] = len(wins)

    # ── P4: short recording ──────────────────────────────────────────────
    if rec['n_windows'] < 2:
        rec['problems'].append(f'P4-too-short ({rec["n_windows"]} window)')

    # ── extract features ─────────────────────────────────────────────────
    feats = np.array([extract_features(w) for w in wins])
    feats = np.nan_to_num(feats, nan=0., posinf=0., neginf=0.)
    rec['feat_mat'] = feats

    # ── P6: NaN / Inf before nan_to_num ─────────────────────────────────
    raw_feats = np.array([extract_features(w) for w in wins])
    if not np.isfinite(raw_feats).all():
        n_bad = (~np.isfinite(raw_feats)).sum()
        rec['problems'].append(f'P6-nan-inf ({n_bad} values)')

    # ── P1: tap detection failure ────────────────────────────────────────
    tap_rates = feats[:, 1]   # dist_tap_rate
    rec['tap_rate_mean']      = float(np.mean(tap_rates))
    rec['n_windows_zero_tap'] = int((tap_rates == 0.).sum())

    if rec['n_windows_zero_tap'] == rec['n_windows']:
        rec['problems'].append('P1-tap-detection-FAILED (0 taps in ALL windows)')
    elif rec['n_windows_zero_tap'] > rec['n_windows'] * 0.5:
        rec['problems'].append(
            f'P1-tap-detection-POOR '
            f'({rec["n_windows_zero_tap"]}/{rec["n_windows"]} windows have 0 taps)')

    # ── P7: implausible tap rate ─────────────────────────────────────────
    valid_rates = tap_rates[tap_rates > 0]
    if len(valid_rates) > 0:
        if np.mean(valid_rates) < 0.3:
            rec['problems'].append(
                f'P7-tap-rate-too-low (mean={np.mean(valid_rates):.2f} Hz)')
        if np.mean(valid_rates) > 6.0:
            rec['problems'].append(
                f'P7-tap-rate-too-high (mean={np.mean(valid_rates):.2f} Hz)')

    return rec


print("Scanning all files …\n")
all_recs = []
for cls in [0, 1]:
    folder = os.path.join(DATA_DIR, str(cls))
    fnames = sorted(f for f in os.listdir(folder) if f.endswith('.txt'))
    for fname in fnames:
        r = check_file(cls, fname)
        all_recs.append(r)

recs0 = [r for r in all_recs if r['cls']==0]
recs1 = [r for r in all_recs if r['cls']==1]

# ════════════════════════════════════════════════════════════════════════════
#  DETECT CROSS-CLASS FEATURE OUTLIERS (P5)
# ════════════════════════════════════════════════════════════════════════════

# Pool per-window features per class
F0 = np.vstack([r['feat_mat'] for r in recs0 if r['feat_mat'] is not None])
F1 = np.vstack([r['feat_mat'] for r in recs1 if r['feat_mat'] is not None])
Fcat = np.vstack([F0, F1])
global_mu  = np.nanmean(Fcat, axis=0)
global_sig = np.nanstd(Fcat,  axis=0)
global_sig[global_sig < 1e-8] = 1.

for r in all_recs:
    if r['feat_mat'] is None: continue
    Z = np.abs((r['feat_mat'] - global_mu) / global_sig)
    bad_wins  = (Z > 6).any(axis=1)
    if bad_wins.any():
        bad_feats = [FEAT_NAMES[i]
                     for i in range(len(FEAT_NAMES)) if (Z[:,i]>6).any()]
        r['problems'].append(
            f'P5-outlier ({bad_wins.sum()}/{r["n_windows"]} windows, '
            f'features: {bad_feats})')

# ════════════════════════════════════════════════════════════════════════════
#  WRITE TEXT REPORT
# ════════════════════════════════════════════════════════════════════════════

report_path = os.path.join(OUT_DIR, 'quality_report.txt')
with open(report_path, 'w') as rpt:
    def w(s=''):  print(s); rpt.write(s+'\n')

    w('=' * 72)
    w('DATA QUALITY REPORT')
    w('=' * 72)

    total = len(all_recs)
    n_clean = sum(1 for r in all_recs if not r['problems'])
    n_prob  = total - n_clean
    w(f'Total files scanned : {total}  (class 0: {len(recs0)}, class 1: {len(recs1)})')
    w(f'Clean files         : {n_clean}')
    w(f'Files with problems : {n_prob}\n')

    # ── problem counts ───────────────────────────────────────────────────
    from collections import Counter
    problem_codes = Counter()
    for r in all_recs:
        for p in r['problems']:
            problem_codes[p.split('-')[0]] += 1
    w('Problem summary:')
    for code, cnt in sorted(problem_codes.items()):
        w(f'  {code}: {cnt} file(s)')
    w()

    # ── per-file details ─────────────────────────────────────────────────
    w('-' * 72)
    w('FILES WITH PROBLEMS:')
    w('-' * 72)
    for r in sorted(all_recs, key=lambda x: (not bool(x['problems']), x['cls'], x['fname'])):
        if not r['problems']:
            continue
        w(f"\n  [CLASS {r['cls']}]  {r['fname']}")
        w(f"    duration={r['duration_s']:.1f}s  windows={r['n_windows']}  "
          f"dist_range={r['dist_range']:.2f}cm  "
          f"tap_rate_mean={r['tap_rate_mean']:.2f}Hz  "
          f"zero_tap_wins={r['n_windows_zero_tap']}")
        for p in r['problems']:
            w(f"    >>> {p}")

    # ── recommended removals ─────────────────────────────────────────────
    w()
    w('=' * 72)
    w('RECOMMENDED REMOVALS (severe problems only):')
    w('  Criteria: P0, P1-FAILED, P2, P3, P4, P6, P7')
    w('=' * 72)
    severe_keywords = ['P0','P1-tap-detection-FAILED','P2','P3','P4','P6','P7']
    remove_0, remove_1 = [], []
    for r in all_recs:
        severe = [p for p in r['problems']
                  if any(p.startswith(k) for k in severe_keywords)]
        if severe:
            if r['cls']==0: remove_0.append((r['fname'], severe))
            else:           remove_1.append((r['fname'], severe))

    w(f"\nFrom class 0 ({len(remove_0)} files):")
    for fn, ps in remove_0:
        w(f"  {fn}")
        for p in ps: w(f"    > {p}")

    w(f"\nFrom class 1 ({len(remove_1)} files):")
    for fn, ps in remove_1:
        w(f"  {fn}")
        for p in ps: w(f"    > {p}")

    # ── borderline (P1-POOR, P5 only) ───────────────────────────────────
    w()
    w('=' * 72)
    w('BORDERLINE (P1-POOR or P5 outliers only — review manually):')
    w('=' * 72)
    borderline_kw = ['P1-tap-detection-POOR', 'P5']
    for r in all_recs:
        has_border = [p for p in r['problems']
                      if any(p.startswith(k) for k in borderline_kw)]
        is_severe  = [p for p in r['problems']
                      if any(p.startswith(k) for k in severe_keywords)]
        if has_border and not is_severe:
            w(f"  [CLASS {r['cls']}] {r['fname']}")
            for p in has_border: w(f"    > {p}")

print(f"\nReport saved: {report_path}")

# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: tap-rate distribution per file  (bar coloured by zero-tap %)
# ════════════════════════════════════════════════════════════════════════════

def plot_tap_rate_overview(recs0, recs1):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle('Per-Recording Tap Rate & Zero-Tap Window Ratio\n'
                 '(red bars = tap-detection FAILED in ≥50% windows)',
                 fontsize=13, fontweight='bold')

    for ax, recs, cls in [(axes[0],recs0,0),(axes[1],recs1,1)]:
        names = [r['fname'][:35] for r in recs]
        rates = [r['tap_rate_mean'] for r in recs]
        zero_frac = [r['n_windows_zero_tap']/max(r['n_windows'],1) for r in recs]
        colors = ['#e74c3c' if zf>=0.5 else '#3498db' for zf in zero_frac]

        x = np.arange(len(names))
        bars = ax.bar(x, rates, color=colors, alpha=0.8, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_ylabel('Mean tap rate (Hz)', fontsize=9)
        ax.set_title(f'UPDRS={cls}', fontsize=10)
        ax.axhline(0.3, color='orange', ls='--', lw=1.2, label='min 0.3 Hz')
        ax.axhline(6.0, color='purple', ls='--', lw=1.2, label='max 6.0 Hz')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        # annotate zero-tap % above bars
        for xi, zf in zip(x, zero_frac):
            if zf > 0:
                ax.text(xi, rates[xi]+0.02, f'{zf:.0%}',
                        ha='center', va='bottom', fontsize=5,
                        color='red' if zf>=0.5 else 'gray')

    fig.tight_layout(rect=[0,0,1,0.95])
    p = os.path.join(OUT_DIR,'fig_quality_taprate.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {p}")


plot_tap_rate_overview(recs0, recs1)


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: dist signal for the worst files
# ════════════════════════════════════════════════════════════════════════════

def plot_bad_dist_signals(all_recs):
    # Collect files with P1-FAILED or P2
    bad = [r for r in all_recs
           if any(p.startswith('P1-tap-detection-FAILED') or
                  p.startswith('P2') for p in r['problems'])]
    if not bad:
        print("No P1-FAILED or P2 files found — skipping fig_quality_dist.")
        return

    n = min(len(bad), 12)
    bad = bad[:n]
    ncols = 4; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4*nrows))
    fig.suptitle('dist Signal of Problematic Files\n'
                 '(P1 = tap detection failed, P2 = near-zero distance)',
                 fontsize=12, fontweight='bold')
    axes_flat = axes.flatten() if nrows > 1 else axes

    for i, r in enumerate(bad):
        ax = axes_flat[i]
        fp = os.path.join(DATA_DIR, str(r['cls']), r['fname'])
        s1p,s2p,s1o,s2o,s1t,s2t = parse_raw(fp)
        data = build_6ch(s1p,s1o,s1t,s2p,s2o,s2t)
        if data is None:
            ax.text(0.5,0.5,'build failed', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        data[:,0] = lowpass(data[:,0])
        t = np.arange(len(data)) / FS
        ax.plot(t, data[:,0], color='tab:blue', lw=0.8)
        ax.set_title(f'CLS {r["cls"]} — {r["fname"][:30]}\n'
                     f'{r["problems"][0][:55]}',
                     fontsize=7)
        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('dist (cm)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(rect=[0,0,1,0.95])
    p = os.path.join(OUT_DIR,'fig_quality_bad_dist.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {p}")


plot_bad_dist_signals(all_recs)


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: per-file feature heatmap (tap_rate, iti_mean, dist_range)
# ════════════════════════════════════════════════════════════════════════════

def plot_recording_heatmap(all_recs):
    # show 3 key diagnostics per recording: tap_rate_mean, iti_mean_mean, dist_range
    fig, axes = plt.subplots(3, 1, figsize=(22, 13))
    fig.suptitle('Per-Recording Diagnostic Heatmap\n'
                 'Red = likely problematic', fontsize=13, fontweight='bold')

    def make_row(recs, cls):
        names = [r['fname'][:32] for r in recs]
        tr    = [r['tap_rate_mean'] for r in recs]
        dr    = [r['dist_range']    for r in recs]
        im    = [np.mean(r['feat_mat'][:,0]) if r['feat_mat'] is not None else 0.
                 for r in recs]
        has_p = [bool(r['problems']) for r in recs]
        return names, np.array(tr), np.array(dr), np.array(im), has_p

    for ax_idx, (recs, cls) in enumerate([(recs0,0),(recs1,1)]):
        names, tr, dr, im, has_p = make_row(recs, cls)

        metrics = np.vstack([tr, dr, im]).T   # (n_files, 3)
        # z-score each column for display
        for col in range(metrics.shape[1]):
            m,s = np.mean(metrics[:,col]),np.std(metrics[:,col])
            if s>1e-8: metrics[:,col] = (metrics[:,col]-m)/s

        pass  # use individual subplots below

    metric_labels = ['tap_rate_mean (Hz)', 'dist_range (cm)', 'iti_mean_mean (s)']
    all_grouped = [(recs0, 0), (recs1, 1)]

    for m_idx, mlabel in enumerate(metric_labels):
        ax = axes[m_idx]
        offset = 0
        xtick_pos, xtick_labels, vlines = [], [], []
        colors = []

        for recs, cls in all_grouped:
            for r in recs:
                if m_idx == 0:   val = r['tap_rate_mean']
                elif m_idx == 1: val = r['dist_range']
                else:            val = (np.mean(r['feat_mat'][:,0])
                                        if r['feat_mat'] is not None else 0.)
                clr = '#e74c3c' if r['problems'] else ('#3498db' if cls==0 else '#e67e22')
                colors.append(clr)
                xtick_pos.append(offset)
                xtick_labels.append(r['fname'][:25])
                ax.bar(offset, val, color=clr, alpha=0.8, width=0.8)
                offset += 1
            vlines.append(offset - 0.5)

        for vl in vlines[:-1]:
            ax.axvline(vl, color='black', lw=1.5, ls='-')
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, rotation=90, fontsize=5)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#e74c3c', label='Has problem'),
            Patch(color='#3498db', label='Clean (cls 0)'),
            Patch(color='#e67e22', label='Clean (cls 1)'),
        ], fontsize=7, loc='upper right')
        # annotation for class boundary
        boundary = len(recs0) - 0.5
        ax.axvline(boundary, color='green', lw=2, ls='--')
        ax.text(boundary-0.5, ax.get_ylim()[1]*0.95, 'cls 0 | cls 1',
                ha='right', fontsize=8, color='green')

    fig.tight_layout(rect=[0,0,1,0.95])
    p = os.path.join(OUT_DIR,'fig_quality_recording_heatmap.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {p}")


plot_recording_heatmap(all_recs)


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: per-window tap-rate strip for all files
# ════════════════════════════════════════════════════════════════════════════

def plot_window_strip(all_recs):
    """Horizontal strip: each column = one window, colour = tap_rate."""
    fig, axes = plt.subplots(2, 1, figsize=(22, 8))
    fig.suptitle('Per-Window Tap Rate (each column = one 5s window)\n'
                 'Black = 0 Hz (tap detection failed)',
                 fontsize=12, fontweight='bold')

    for ax, recs, cls in [(axes[0],recs0,0),(axes[1],recs1,1)]:
        mat_rows = []
        yticks   = []
        for r in recs:
            if r['feat_mat'] is None: continue
            mat_rows.append(r['feat_mat'][:,1])   # dist_tap_rate
            yticks.append(r['fname'][:30])

        if not mat_rows: continue
        max_w = max(len(m) for m in mat_rows)
        M = np.full((len(mat_rows), max_w), np.nan)
        for i, row in enumerate(mat_rows):
            M[i, :len(row)] = row

        im = ax.imshow(M, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=4, interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.6, label='tap rate (Hz)')
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=5.5)
        ax.set_xlabel('Window index', fontsize=9)
        ax.set_title(f'UPDRS={cls}', fontsize=10)

    fig.tight_layout(rect=[0,0,1,0.95])
    p = os.path.join(OUT_DIR,'fig_quality_window_strip.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"Saved: {p}")


plot_window_strip(all_recs)

# ════════════════════════════════════════════════════════════════════════════
#  FINAL COUNTS
# ════════════════════════════════════════════════════════════════════════════

severe_kw    = ['P0','P1-tap-detection-FAILED','P2','P3','P4','P6','P7']
remove_0_cnt = sum(1 for r in recs0 if any(p.startswith(k)
                   for k in severe_kw for p in r['problems']))
remove_1_cnt = sum(1 for r in recs1 if any(p.startswith(k)
                   for k in severe_kw for p in r['problems']))

print('\n' + '='*65)
print('FINAL SUMMARY')
print('='*65)
print(f'  Class 0: {len(recs0)} files  →  {remove_0_cnt} recommended for removal')
print(f'  Class 1: {len(recs1)} files  →  {remove_1_cnt} recommended for removal')
print(f'  After removal: class 0 = {len(recs0)-remove_0_cnt}, '
      f'class 1 = {len(recs1)-remove_1_cnt}')
print(f'\n  Full report: {report_path}')
print(f'  Figures:     {OUT_DIR}/')
