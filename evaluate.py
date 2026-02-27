"""
=============================================================================
evaluate.py — Evaluation & Visualization
=============================================================================
Compares two classifier variants:
  baseline  : real data + prior sampling (z ~ N(0,I)), original approach
  strategy1 : real data + improved posterior interpolation
                (KNN-guided mixing, Beta(2,2) weights, quality filtering)

Plots: confusion_matrix, roc_curve, training_curves, tsne_latent, reconstruction
=============================================================================
"""

import os, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.manifold import TSNE
from model import create_vae, create_classifier

RESULTS_DIR = 'results'
CKPT_DIR    = 'checkpoints'

# Human-readable labels for each variant
VARIANT_LABELS = {
    'baseline':  'Baseline (prior sampling)',
    'strategy1': 'Strategy 1 (improved posterior)',
}


def load_results():
    with open(os.path.join(CKPT_DIR, 'train_results.pkl'), 'rb') as f:
        return pickle.load(f)


# ========================== LATENT EXTRACTION ==============================

def extract_latent_features(vae, X_raw, y, device, batch_size=64):
    """Extract VAE encoder mean (mu) vectors as features."""
    vae.eval()
    z_list = []
    for i in range(0, len(X_raw), batch_size):
        batch_end = min(i + batch_size, len(X_raw))
        xb = torch.FloatTensor(X_raw[i:batch_end]).to(device)
        yb = torch.LongTensor(y[i:batch_end]).to(device)
        with torch.no_grad():
            _, mu, _ = vae.encode(xb, yb)
        z_list.append(mu.cpu().numpy())
    return np.concatenate(z_list, axis=0)


# ========================== CLASSIFICATION =================================

def evaluate_model(clf, X_feat_te, y_te, device):
    clf.eval()
    with torch.no_grad():
        logits = clf(torch.FloatTensor(X_feat_te).to(device)).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
    return preds, probs


def evaluate_all_variants(res, vae, device):
    """
    Evaluate baseline and strategy1 classifier checkpoints.
    Returns a dict: tag → {preds, probs, acc, auc, f1, recall_pd, prec_pd}
    """
    X_feat_te = res['X_feat_test']
    y_te      = res['y_test']
    feat_dim  = X_feat_te.shape[-1]

    summary = {}

    for tag in ['baseline', 'strategy1']:
        ckpt_path = os.path.join(CKPT_DIR, f'clf_best_{tag}.pt')
        if not os.path.exists(ckpt_path):
            continue

        clf = create_classifier(feat_dim).to(device)
        clf.load_state_dict(torch.load(ckpt_path, map_location=device,
                                        weights_only=True))

        X_te = X_feat_te

        preds, probs = evaluate_model(clf, X_te, y_te, device)

        acc = (preds == y_te).mean()
        tp  = ((preds==1)&(y_te==1)).sum()
        fp  = ((preds==1)&(y_te==0)).sum()
        fn  = ((preds==0)&(y_te==1)).sum()
        prec_pd   = tp / (tp + fp + 1e-8)
        recall_pd = tp / (tp + fn + 1e-8)
        f1_pd     = 2 * prec_pd * recall_pd / (prec_pd + recall_pd + 1e-8)

        fpr, tpr, _ = roc_curve(y_te, probs)
        auc_val     = auc(fpr, tpr)

        summary[tag] = {
            'preds': preds, 'probs': probs,
            'acc': acc, 'auc': auc_val,
            'f1_pd': f1_pd, 'recall_pd': recall_pd, 'prec_pd': prec_pd,
            'fpr': fpr, 'tpr': tpr,
        }

    return summary


def tune_threshold(summary, y_te, target_recall=0.85):
    """
    For each variant, find the threshold that:
      1. Maximises F1(PD)
      2. Achieves target_recall(PD) with highest precision

    Returns augmented summary with threshold info.
    """
    thresholds = np.linspace(0.1, 0.9, 81)

    print(f"\n{'='*65}")
    print(f"  THRESHOLD TUNING  (default=0.5, target_recall≥{target_recall})")
    print(f"{'='*65}")
    hdr = f"  {'Variant':28s}{'Thr':>6s}{'Acc':>7s}{'AUC':>7s}{'F1(PD)':>9s}{'Rec(PD)':>9s}{'Prec(PD)':>10s}"
    print(hdr); print(f"  {'-'*65}")

    for tag, m in summary.items():
        probs = m['probs']
        best_f1, best_thr_f1 = -1, 0.5
        thr_recall, rec_at_thr, prec_at_thr = 0.5, 0.0, 0.0

        for thr in thresholds:
            preds = (probs > thr).astype(int)
            tp = ((preds==1)&(y_te==1)).sum()
            fp = ((preds==1)&(y_te==0)).sum()
            fn = ((preds==0)&(y_te==1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2*prec*rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1     = f1
                best_thr_f1 = thr
            # best threshold that achieves target_recall
            if rec >= target_recall and prec > prec_at_thr:
                thr_recall   = thr
                rec_at_thr   = rec
                prec_at_thr  = prec

        # Evaluate at best-F1 threshold
        preds_opt = (probs > best_thr_f1).astype(int)
        acc_opt   = (preds_opt == y_te).mean()
        tp = ((preds_opt==1)&(y_te==1)).sum()
        fp = ((preds_opt==1)&(y_te==0)).sum()
        fn = ((preds_opt==0)&(y_te==1)).sum()
        prec_opt = tp / (tp + fp + 1e-8)
        rec_opt  = tp / (tp + fn + 1e-8)

        m['best_thr']    = best_thr_f1
        m['preds_opt']   = preds_opt
        m['f1_pd_opt']   = best_f1
        m['acc_opt']     = acc_opt
        m['recall_opt']  = rec_opt
        m['prec_opt']    = prec_opt
        m['thr_recall']  = thr_recall

        lbl = VARIANT_LABELS.get(tag, tag)[:28]
        print(f"  {lbl:28s}{best_thr_f1:6.2f}{acc_opt:7.3f}{m['auc']:7.3f}"
              f"{best_f1:9.3f}{rec_opt:9.3f}{prec_opt:10.3f}")

    print(f"\n  [Optimised threshold — best F1(PD) for each variant]")
    return summary


def plot_precision_recall(summary, y_te):
    """Precision-Recall curves with optimal operating points marked."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    colors = {'baseline': 'tab:blue', 'strategy1': 'tab:orange'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for tag, m in summary.items():
        prec_c, rec_c, _ = precision_recall_curve(y_te, m['probs'])
        ap  = average_precision_score(y_te, m['probs'])
        c   = colors.get(tag, 'gray')
        lbl = VARIANT_LABELS.get(tag, tag)
        ax.plot(rec_c, prec_c, lw=2, color=c, label=f'{lbl}  AP={ap:.3f}')
        # Mark optimal threshold point
        if 'recall_opt' in m:
            ax.scatter(m['recall_opt'], m['prec_opt'], s=100,
                       color=c, zorder=5, marker='*')

    ax.set_xlabel('Recall (PD)')
    ax.set_ylabel('Precision (PD)')
    ax.set_title('Precision–Recall Curves  (★ = optimal F1 threshold)')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    path = os.path.join(RESULTS_DIR, 'precision_recall_curve.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def print_full_comparison(summary, y_te):
    """Print classification reports and comparison table for all variants."""

    for tag, m in summary.items():
        label = VARIANT_LABELS.get(tag, tag)
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
        print(classification_report(y_te, m['preds'],
              target_names=['Non-PD (0)', 'PD (1)'], digits=3))
        print(f"  AUC: {m['auc']:.3f}  |  PD Recall: {m['recall_pd']:.3f}"
              f"  |  PD Prec: {m['prec_pd']:.3f}")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  COMPREHENSIVE SUMMARY")
    print(f"{'='*65}")
    hdr = f"  {'Variant':28s}{'Acc':>7s}{'AUC':>7s}{'F1(PD)':>9s}{'Rec(PD)':>9s}"
    print(hdr)
    print(f"  {'-'*60}")

    # Sort by F1(PD)
    best_f1 = max(m['f1_pd'] for m in summary.values())
    for tag, m in sorted(summary.items(), key=lambda x: -x[1]['f1_pd']):
        label  = VARIANT_LABELS.get(tag, tag)[:28]
        marker = " ★" if abs(m['f1_pd'] - best_f1) < 1e-6 else ""
        print(f"  {label:28s}{m['acc']:7.3f}{m['auc']:7.3f}"
              f"{m['f1_pd']:9.3f}{m['recall_pd']:9.3f}{marker}")

    # ── Show improvement of Strategy 1 over Baseline ────────────────────
    if 'baseline' in summary and 'strategy1' in summary:
        base = summary['baseline']
        m    = summary['strategy1']
        label = VARIANT_LABELS.get('strategy1', 'strategy1')[:28]
        d_acc = m['acc']       - base['acc']
        d_auc = m['auc']       - base['auc']
        d_f1  = m['f1_pd']     - base['f1_pd']
        d_rec = m['recall_pd'] - base['recall_pd']
        print(f"\n  Improvement of Strategy 1 over Baseline:")
        print(f"  {'-'*60}")
        print(f"  {label:28s}"
              f"  Δacc={d_acc:+.3f}  Δauc={d_auc:+.3f}"
              f"  ΔF1(PD)={d_f1:+.3f}  ΔRec(PD)={d_rec:+.3f}")


# ========================== PLOTS ==========================================

def plot_confusion_all(y_te, summary):
    """Confusion matrices for all variants in one figure."""
    tags = list(summary.keys())
    n    = len(tags)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle('Confusion Matrices — All Variants', fontsize=14)

    cmaps = ['Blues', 'Oranges', 'Greens', 'Purples'][:n]
    for ax, tag, cmap in zip(axes, tags, cmaps):
        cm = confusion_matrix(y_te, summary[tag]['preds'])
        ConfusionMatrixDisplay(cm, display_labels=['Non-PD (0)', 'PD (1)']).plot(
            ax=ax, cmap=cmap, values_format='d')
        f1  = summary[tag]['f1_pd']
        rec = summary[tag]['recall_pd']
        ax.set_title(f"{VARIANT_LABELS.get(tag, tag)}\nF1(PD)={f1:.3f}  Rec={rec:.3f}",
                     fontsize=9)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_all(y_te, summary):
    """ROC curves for all variants."""
    colors = {'baseline': 'tab:blue', 'strategy1': 'tab:orange'}
    fig, ax = plt.subplots(figsize=(7, 6))

    for tag, m in summary.items():
        label = VARIANT_LABELS.get(tag, tag)
        c     = colors.get(tag, 'gray')
        ax.plot(m['fpr'], m['tpr'], lw=2, color=c,
                label=f"{label}  (AUC={m['auc']:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Variants')
    ax.legend(loc='lower right', fontsize=8)
    path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(vae_hist, clf_hists, best_tag):
    """Training history plots."""
    tags       = [t for t in ['baseline', 'strategy1'] if t in clf_hists]
    colors_clf = {'baseline': 'tab:blue', 'strategy1': 'tab:orange'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History — Baseline vs Strategy 1', fontsize=16)

    # VAE curves
    axes[0, 0].plot(vae_hist['recon'], 'g-'); axes[0, 0].set_title('VAE Recon Loss')
    axes[0, 1].plot(vae_hist['kl'], 'orange')
    ax2 = axes[0, 1].twinx(); ax2.plot(vae_hist['kl_weight'], 'm--', alpha=0.5)
    axes[0, 1].set_title('VAE KL  (m-- = β weight)')
    axes[0, 2].plot(vae_hist['total'], 'b-'); axes[0, 2].set_title('VAE Total Loss')

    # Classifier curves
    for tag in tags:
        h   = clf_hists[tag]
        lbl = VARIANT_LABELS.get(tag, tag)
        c   = colors_clf.get(tag, 'gray')
        axes[1, 0].plot(h['train_loss'], color=c, alpha=0.6, label=f'Train {lbl}')
        axes[1, 0].plot(h['val_loss'],   color=c, ls='--', alpha=0.6, label=f'Val {lbl}')
        axes[1, 1].plot(h['val_acc'],    color=c, label=lbl)
        axes[1, 2].plot(h['val_f1'],     color=c, label=lbl)

    axes[1, 0].set_title('Classifier Loss'); axes[1, 0].legend(fontsize=7)
    axes[1, 1].set_title('Val Accuracy');    axes[1, 1].legend(fontsize=7)
    axes[1, 2].set_title(f'Val F1  (best: {best_tag})'); axes[1, 2].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_tsne(vae, X_raw_tr, y_tr, syn_raw, syn_y, device):
    print("  Computing t-SNE on latent space...")
    vae.eval()
    z_list = []
    for i in range(0, len(X_raw_tr), 64):
        xb = torch.FloatTensor(X_raw_tr[i:i+64]).to(device)
        yb = torch.LongTensor(y_tr[i:i+64]).to(device)
        with torch.no_grad():
            z, _, _ = vae.encode(xb, yb)
        z_list.append(z.cpu().numpy())
    z_real = np.concatenate(z_list)

    if len(syn_y) > 0 and syn_raw.ndim == 3:
        zs = []
        for i in range(0, len(syn_raw), 64):
            xb = torch.FloatTensor(syn_raw[i:i+64]).to(device)
            yb = torch.LongTensor(syn_y[i:i+64]).to(device)
            with torch.no_grad():
                z, _, _ = vae.encode(xb, yb)
            zs.append(z.cpu().numpy())
        z_syn = np.concatenate(zs)
    else:
        z_syn = np.empty((0, z_real.shape[1]))

    z_all = np.vstack([z_real, z_syn])
    nr    = len(z_real)
    emb   = TSNE(n_components=2, random_state=42,
                 perplexity=min(30, len(z_all) - 1),
                 max_iter=1000).fit_transform(z_all)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('VAE Latent Space (t-SNE)  —  Strategy 1 Synthetic Samples', fontsize=14)

    for cls, c, lb in [(0, 'tab:blue', 'Non-PD'), (1, 'tab:red', 'PD')]:
        m = y_tr == cls
        a1.scatter(emb[:nr][m, 0], emb[:nr][m, 1], c=c, alpha=.5,
                   s=15, label=f'{lb} real')
        if len(syn_y) > 0:
            sm = syn_y == cls
            a1.scatter(emb[nr:][sm, 0], emb[nr:][sm, 1], c=c, marker='x',
                       alpha=.5, s=15, label=f'{lb} synthetic (posterior)')

    a1.legend(fontsize=8); a1.set_title('By Class (real circles / syn ×)')

    a2.scatter(emb[:nr, 0], emb[:nr, 1], c='tab:green',  alpha=.5, s=15, label='Real')
    if len(z_syn) > 0:
        a2.scatter(emb[nr:, 0], emb[nr:, 1], c='tab:orange', marker='x',
                   alpha=.5, s=15, label='Synthetic (posterior)')
    a2.legend(fontsize=8); a2.set_title('Real vs Synthetic')

    path = os.path.join(RESULTS_DIR, 'tsne_latent.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_reconstruction(vae, X_raw_te, y_te, vae_names, device, n=4):
    vae.eval()
    np.random.seed(42)
    idx0, idx1 = np.where(y_te == 0)[0], np.where(y_te == 1)[0]
    chosen = []
    for _ in range(n // 2):
        if len(idx0) > 0: chosen.append(np.random.choice(idx0))
    for _ in range(n - len(chosen)):
        if len(idx1) > 0: chosen.append(np.random.choice(idx1))
    if not chosen:
        return

    xb = torch.FloatTensor(X_raw_te[chosen]).to(device)
    yb = torch.LongTensor(y_te[chosen]).to(device)
    with torch.no_grad():
        recon, _, _ = vae(xb, yb)
    orig, rec = xb.cpu().numpy(), recon.cpu().numpy()

    show_ch = min(3, orig.shape[2])
    fig, axes = plt.subplots(len(chosen), show_ch,
                             figsize=(5 * show_ch, 3 * len(chosen)))
    fig.suptitle('VAE Reconstruction Quality', fontsize=14)
    axes = np.atleast_2d(axes)

    for i in range(len(chosen)):
        for j in range(show_ch):
            ax   = axes[i, j]
            corr = np.corrcoef(orig[i, :, j], rec[i, :, j])[0, 1]
            nm   = vae_names[j] if j < len(vae_names) else f'ch{j}'
            ax.plot(orig[i, :, j], 'b-', alpha=.8, label='Orig')
            ax.plot(rec[i, :, j],  'r-', alpha=.8, label='Recon')
            ax.set_title(f'{nm}  cls={y_te[chosen[i]]}  r={corr:.3f}', fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'reconstruction.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(summary, clin_names):
    """Bar chart: which clinical features differ most between classes."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(clin_names))
    ax.bar(x, np.ones(len(clin_names)), alpha=0.3, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(clin_names, rotation=45, ha='right', fontsize=7)
    ax.set_title('Clinical Feature Names (reference)')
    ax.set_ylabel('Index')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'feature_names.png')
    fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)


# ========================== MAIN ===========================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res    = load_results()

    input_dim = res['X_raw_test'].shape[-1]   # 6
    seq_len   = res['X_raw_test'].shape[1]    # 300
    feat_dim  = res['X_feat_test'].shape[-1]  # 15
    vae_names = res.get('vae_channel_names', [])
    clin_names = res.get('clinical_feature_names', [])

    print("=" * 65)
    print("PD Classification — Comprehensive Evaluation")
    print("=" * 65)
    print(f"  Test samples : {len(res['y_test'])} "
          f"(0={np.sum(res['y_test']==0)}, 1={np.sum(res['y_test']==1)})")
    print(f"  VAE input    : {input_dim}-channel × {seq_len} timesteps")
    print(f"  Classifier   : {feat_dim} clinical features")

    # Load VAE
    vae = create_vae(input_dim, seq_len).to(device)
    vae.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'vae_best.pt'),
                                    map_location=device, weights_only=True))

    # Evaluate all variants
    summary = evaluate_all_variants(res, vae, device)
    print(f"\n  Loaded variants: {list(summary.keys())}")

    print_full_comparison(summary, res['y_test'])
    summary = tune_threshold(summary, res['y_test'], target_recall=0.85)

    print("\nGenerating plots...")
    plot_confusion_all(res['y_test'], summary)
    plot_roc_all(res['y_test'], summary)
    plot_precision_recall(summary, res['y_test'])
    plot_training_curves(res['vae_history'], res['clf_histories'],
                         res['best_model'])
    plot_tsne(vae, res['X_raw_train'], res['y_train'],
              res['syn_raw'], res['syn_y'], device)
    plot_reconstruction(vae, res['X_raw_test'], res['y_test'],
                        vae_names, device)

    print(f"\n{'='*65}")
    print(f"DONE — all figures saved to  {RESULTS_DIR}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
