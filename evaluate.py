"""
=============================================================================
evaluate.py — Evaluation & Visualization
=============================================================================
Before VAE (real-only) vs After VAE (real+synthetic) comparison.
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
CKPT_DIR = 'checkpoints'


def load_results():
    with open(os.path.join(CKPT_DIR, 'train_results.pkl'), 'rb') as f:
        return pickle.load(f)


# ========================== CLASSIFICATION =================================

def evaluate_model(clf, X_feat_te, y_te, device):
    clf.eval()
    with torch.no_grad():
        logits = clf(torch.FloatTensor(X_feat_te).to(device)).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    return preds, probs


def print_comparison(X_feat_te, y_te, feat_dim, device):
    clf_real = create_classifier(feat_dim).to(device)
    clf_real.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'clf_best_real.pt'),
        map_location=device, weights_only=True))

    clf_aug = create_classifier(feat_dim).to(device)
    aug_path = os.path.join(CKPT_DIR, 'clf_best_aug.pt')
    has_aug = os.path.exists(aug_path)
    if has_aug:
        clf_aug.load_state_dict(torch.load(
            aug_path, map_location=device, weights_only=True))

    preds_real, probs_real = evaluate_model(clf_real, X_feat_te, y_te, device)

    print("\n" + "=" * 60)
    print("BEFORE VAE (Real Data Only)")
    print("=" * 60)
    print(classification_report(y_te, preds_real,
          target_names=['Non-PD (0)', 'PD (1)'], digits=3))
    fpr_r, tpr_r, _ = roc_curve(y_te, probs_real)
    auc_real = auc(fpr_r, tpr_r)
    print(f"  AUC: {auc_real:.3f}")

    if has_aug:
        preds_aug, probs_aug = evaluate_model(clf_aug, X_feat_te, y_te, device)
        print("\n" + "=" * 60)
        print("AFTER VAE (Real + Synthetic Data)")
        print("=" * 60)
        print(classification_report(y_te, preds_aug,
              target_names=['Non-PD (0)', 'PD (1)'], digits=3))
        fpr_a, tpr_a, _ = roc_curve(y_te, probs_aug)
        auc_aug = auc(fpr_a, tpr_a)
        print(f"  AUC: {auc_aug:.3f}")
    else:
        preds_aug, probs_aug = preds_real, probs_real
        auc_aug = auc_real
        print("\n  [No augmented model found — skipping After VAE]")

    acc_r = (preds_real == y_te).mean()
    acc_a = (preds_aug == y_te).mean()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'':20s} {'Before VAE':>12s}  {'After VAE':>12s}  {'Change':>10s}")
    print(f"  {'Accuracy':20s} {acc_r:>12.3f}  {acc_a:>12.3f}  {acc_a-acc_r:>+10.3f}")
    print(f"  {'AUC':20s} {auc_real:>12.3f}  {auc_aug:>12.3f}  {auc_aug-auc_real:>+10.3f}")

    return preds_real, probs_real, preds_aug, probs_aug


# ========================== PLOTS ==========================================

def plot_confusion(y_te, preds_real, preds_aug):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrix', fontsize=14)

    cm_r = confusion_matrix(y_te, preds_real)
    ConfusionMatrixDisplay(cm_r, display_labels=['Non-PD (0)', 'PD (1)']).plot(
        ax=ax1, cmap='Blues', values_format='d')
    ax1.set_title('Before VAE (Real Only)')

    cm_a = confusion_matrix(y_te, preds_aug)
    ConfusionMatrixDisplay(cm_a, display_labels=['Non-PD (0)', 'PD (1)']).plot(
        ax=ax2, cmap='Oranges', values_format='d')
    ax2.set_title('After VAE (Real + Synthetic)')

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc(y_te, probs_real, probs_aug):
    fpr_r, tpr_r, _ = roc_curve(y_te, probs_real)
    auc_r = auc(fpr_r, tpr_r)
    fpr_a, tpr_a, _ = roc_curve(y_te, probs_aug)
    auc_a = auc(fpr_a, tpr_a)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_r, tpr_r, 'b-', lw=2, label=f'Before VAE (AUC={auc_r:.3f})')
    ax.plot(fpr_a, tpr_a, 'r-', lw=2, label=f'After VAE (AUC={auc_a:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Before vs After VAE')
    ax.legend(loc='lower right')
    path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(vae_hist, clf_hists, best_tag):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16)

    axes[0, 0].plot(vae_hist['recon'], 'g-'); axes[0, 0].set_title('VAE Recon Loss')
    axes[0, 1].plot(vae_hist['kl'], 'orange')
    ax2 = axes[0, 1].twinx(); ax2.plot(vae_hist['kl_weight'], 'm--', alpha=0.5)
    axes[0, 1].set_title('VAE KL')
    axes[0, 2].plot(vae_hist['total'], 'b-'); axes[0, 2].set_title('VAE Total')

    for tag, h in clf_hists.items():
        lbl = 'Before VAE' if tag == 'real' else 'After VAE'
        axes[1, 0].plot(h['train_loss'], label=f'Train ({lbl})', alpha=0.7)
        axes[1, 0].plot(h['val_loss'], '--', label=f'Val ({lbl})', alpha=0.7)
        axes[1, 1].plot(h['val_acc'], label=lbl)
        axes[1, 2].plot(h['val_f1'], label=lbl)
    axes[1, 0].set_title('Classifier Loss'); axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_title('Classifier Val Accuracy'); axes[1, 1].legend()
    axes[1, 2].set_title(f'Classifier Val F1 (best: {best_tag})'); axes[1, 2].legend()

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


def plot_tsne(vae, X_raw_tr, y_tr, syn_raw, syn_y, device):
    print("  Computing t-SNE...")
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
    nr = len(z_real)
    emb = TSNE(n_components=2, random_state=42,
               perplexity=min(30, len(z_all) - 1),
               max_iter=1000).fit_transform(z_all)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('VAE Latent Space (t-SNE)', fontsize=14)
    for cls, c, lb in [(0, 'tab:blue', 'Non-PD'), (1, 'tab:red', 'PD')]:
        m = y_tr == cls
        a1.scatter(emb[:nr][m, 0], emb[:nr][m, 1], c=c, alpha=.5,
                   s=15, label=f'{lb} real')
        if len(syn_y) > 0:
            sm = syn_y == cls
            a1.scatter(emb[nr:][sm, 0], emb[nr:][sm, 1], c=c, marker='x',
                       alpha=.5, s=15, label=f'{lb} syn')
    a1.legend(fontsize=8); a1.set_title('By Class')
    a2.scatter(emb[:nr, 0], emb[:nr, 1], c='tab:green', alpha=.5,
               s=15, label='Real')
    if len(z_syn) > 0:
        a2.scatter(emb[nr:, 0], emb[nr:, 1], c='tab:orange', marker='x',
                   alpha=.5, s=15, label='Synthetic')
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

    # Show first 3 channels: dist, s1_pitch, s1_roll
    show_ch = min(3, orig.shape[2])
    fig, axes = plt.subplots(len(chosen), show_ch,
                             figsize=(5 * show_ch, 3 * len(chosen)))
    fig.suptitle('VAE Reconstruction (6-ch)', fontsize=14)
    axes = np.atleast_2d(axes)
    for i in range(len(chosen)):
        for j in range(show_ch):
            ax = axes[i, j]
            ax.plot(orig[i, :, j], 'b-', alpha=.8, label='Orig')
            ax.plot(rec[i, :, j], 'r-', alpha=.8, label='Recon')
            corr = np.corrcoef(orig[i, :, j], rec[i, :, j])[0, 1]
            nm = vae_names[j] if j < len(vae_names) else f'ch{j}'
            ax.set_title(f'{nm} (cls {y_te[chosen[i]]}) r={corr:.3f}',
                         fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, 'reconstruction.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {path}")


# ========================== MAIN ===========================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = load_results()

    input_dim = res['X_raw_test'].shape[-1]   # 6
    seq_len = res['X_raw_test'].shape[1]      # 300
    feat_dim = res['X_feat_test'].shape[-1]   # 15
    vae_names = res.get('vae_channel_names', [])

    print("=" * 60)
    print("PD Classification — Evaluation")
    print("=" * 60)
    print(f"  Test samples: {len(res['y_test'])} "
          f"(0={np.sum(res['y_test']==0)}, 1={np.sum(res['y_test']==1)})")
    print(f"  VAE: {input_dim}-channel, Classifier: {feat_dim} features")

    vae = create_vae(input_dim, seq_len).to(device)
    vae.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'vae_best.pt'),
                                    map_location=device, weights_only=True))

    preds_real, probs_real, preds_aug, probs_aug = print_comparison(
        res['X_feat_test'], res['y_test'], feat_dim, device)

    print("\nGenerating plots...")
    plot_confusion(res['y_test'], preds_real, preds_aug)
    plot_roc(res['y_test'], probs_real, probs_aug)
    plot_training_curves(res['vae_history'], res['clf_histories'],
                         res['best_model'])
    plot_tsne(vae, res['X_raw_train'], res['y_train'],
              res['syn_raw'], res['syn_y'], device)
    plot_reconstruction(vae, res['X_raw_test'], res['y_test'],
                        vae_names, device)

    print(f"\n{'='*60}")
    print(f"DONE — all figures in {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()