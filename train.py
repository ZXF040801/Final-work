import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_LSTM import (create_vae, create_classifier, vae_loss_fn)
from preprocessing import (
    patient_aware_split, compute_normalization_stats, normalize,
    extract_features_batch
)


class TrainConfig:
    SEED   = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_PATH = 'preprocessed/preprocessed_data.pkl'
    CKPT_DIR  = 'checkpoints'

    # ── Feature-Space LSTM-VAE ────────────────────────────────────────
    VAE_WARMUP_EPOCHS = 200   # Phase 1a: β=0，纯重建
    VAE_ANNEAL_EPOCHS = 400   # Phase 1b: β 从 0 线性升至 KL_MAX
    VAE_BATCH  = 64
    VAE_LR     = 1e-3
    KL_MAX     = 0.005
    FREE_BITS  = 0.1

    # ── Classifier ────────────────────────────────────────────────────
    CLF_EPOCHS   = 1000   # 固定训练轮数，不提前停止
    CLF_BATCH    = 32
    CLF_LR       = 1e-3
    CLF_WD       = 1e-2
    LABEL_SMOOTH = 0.1
    PATIENCE     = 9999   # 关闭早停

    GEN_NOISE = 0.02      # 特征空间添加的微小噪声


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, s=0.1):
        super().__init__()
        self.s = s

    def forward(self, logits, targets):
        targets = targets.float() * (1 - self.s) + 0.5 * self.s
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


# ========================== DATA ===========================================

def load_data(cfg):
    with open(cfg.DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    X_raw, X_feat, y = data['X_raw'], data['X_feat'], data['y']
    pids       = data['patient_ids']
    fids       = data.get('file_ids', None)
    vae_names  = data.get('vae_channel_names', [])
    clin_names = data['clinical_feature_names']

    print(f"[Load] X_raw={X_raw.shape}, X_feat={X_feat.shape}, "
          f"y: {dict(Counter(y))}, patients={len(set(pids))}")

    split = patient_aware_split(X_raw, X_feat, y, pids, fids)
    print(f"[Split] Train: {len(split['y_tr'])} ({dict(Counter(split['y_tr']))})")
    print(f"[Split] Test:  {len(split['y_te'])} ({dict(Counter(split['y_te']))})")

    # Normalize raw 6-ch（保留给 evaluate 重建图备用）
    raw_mean, raw_std = compute_normalization_stats(split['X_raw_tr'])
    split['X_raw_tr'] = normalize(split['X_raw_tr'], raw_mean, raw_std)
    split['X_raw_te'] = normalize(split['X_raw_te'], raw_mean, raw_std)

    # Normalize 15 features（VAE 和 Classifier 共用同一份）
    feat_mean = split['X_feat_tr'].mean(axis=0)
    feat_std  = split['X_feat_tr'].std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    split['X_feat_tr'] = (split['X_feat_tr'] - feat_mean) / feat_std
    split['X_feat_te'] = (split['X_feat_te'] - feat_mean) / feat_std

    return split, raw_mean, raw_std, feat_mean, feat_std, vae_names, clin_names


# ========================== PHASE 1: LSTM-VAE (Feature Space) ==============

def train_vae(X_feat_tr, y_tr, cfg):
    """
    在归一化的 15 维特征空间训练 LSTM-VAE。
    输入 X_feat_tr: (N, 15)，已归一化。
    LSTM 把 15 个特征当作长度为 15 的时间序列处理。
    """
    device       = cfg.DEVICE
    feat_dim     = X_feat_tr.shape[-1]   # 15
    total_epochs = cfg.VAE_WARMUP_EPOCHS + cfg.VAE_ANNEAL_EPOCHS

    vae       = create_vae(feat_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5)

    # 预加载到 device
    X_dev = torch.FloatTensor(X_feat_tr).to(device)
    y_dev = torch.LongTensor(y_tr).to(device)
    loader = DataLoader(
        TensorDataset(X_dev, y_dev),
        batch_size=cfg.VAE_BATCH, shuffle=True)

    print(f"\n{'='*60}")
    print(f"PHASE 1: Training LSTM-VAE on {feat_dim}-dim clinical features")
    print(f"  (15 features treated as a sequence of 15 timesteps × 1 dim)")
    print(f"  Phase 1a: {cfg.VAE_WARMUP_EPOCHS} epochs WARMUP (β=0, pure recon)")
    print(f"  Phase 1b: {cfg.VAE_ANNEAL_EPOCHS} epochs ANNEAL (β: 0 → {cfg.KL_MAX})")
    print(f"  Total: {total_epochs} epochs, batch={cfg.VAE_BATCH}")
    print(f"{'='*60}")

    history    = {'total': [], 'recon': [], 'kl': [], 'kl_weight': []}
    best_recon = float('inf')

    for epoch in range(total_epochs):
        if epoch < cfg.VAE_WARMUP_EPOCHS:
            kl_w  = 0.0
            phase = "warmup"
        else:
            progress = (epoch - cfg.VAE_WARMUP_EPOCHS) / cfg.VAE_ANNEAL_EPOCHS
            kl_w  = cfg.KL_MAX * progress
            phase = "anneal"

        vae.train()
        ep_loss, ep_recon, ep_kl, n = 0, 0, 0, 0
        t0 = time.time()

        for xb, yb in loader:
            recon, mu, lv = vae(xb, yb)
            loss, rl, kl  = vae_loss_fn(recon, xb, mu, lv,
                                         kl_weight=kl_w,
                                         free_bits=cfg.FREE_BITS)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            bs        = xb.size(0)
            ep_loss  += loss.item() * bs
            ep_recon += rl.item()   * bs
            ep_kl    += kl.item()   * bs
            n        += bs

        scheduler.step()
        avg_recon = ep_recon / n
        history['total'].append(ep_loss  / n)
        history['recon'].append(avg_recon)
        history['kl'].append(ep_kl    / n)
        history['kl_weight'].append(kl_w)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  [{phase:6s}] Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Recon: {avg_recon:.4f} | KL: {ep_kl/n:.3f} | "
                  f"β: {kl_w:.5f} | {time.time()-t0:.1f}s")

        if avg_recon < best_recon:
            best_recon = avg_recon
            ckpt_path  = os.path.join(cfg.CKPT_DIR, 'vae_best.pt')
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            torch.save(vae.state_dict(), ckpt_path)

    vae.load_state_dict(torch.load(
        os.path.join(cfg.CKPT_DIR, 'vae_best.pt'),
        map_location=device, weights_only=True))

    # ── 质量检验：per-feature 相关系数 ───────────────────────────────
    vae.eval()
    with torch.no_grad():
        n_check = min(len(X_feat_tr), 256)
        xb = torch.FloatTensor(X_feat_tr[:n_check]).to(device)
        yb = torch.LongTensor(y_tr[:n_check]).to(device)
        recon, _, _ = vae(xb, yb)
        orig = xb.cpu().numpy()
        rec  = recon.cpu().numpy()

    corrs = [np.corrcoef(orig[:, i], rec[:, i])[0, 1]
             for i in range(feat_dim)]
    print(f"\n  [VAE Quality] Per-feature correlation")
    print(f"    Overall mean r = {np.nanmean(corrs):.3f}, "
          f"min r = {np.nanmin(corrs):.3f}")
    for i, r in enumerate(corrs):
        print(f"    feat{i:02d}: r={r:.3f}")

    return vae, history


# ========================== PHASE 2: GENERATE FEATURES ====================

def generate_synthetic(vae, y_tr, cfg):
    """
    直接在归一化特征空间生成少数类样本。
    无需生成原始信号，也无需重新提取特征。
    返回 syn_feat (N, 15) 已归一化，syn_y (N,)
    """
    device         = cfg.DEVICE
    counts         = Counter(y_tr)
    majority_count = max(counts.values())

    print(f"\n{'='*60}")
    print(f"PHASE 2: Generating Synthetic Features (balance to {majority_count})")
    print(f"  Class counts: {dict(counts)}")
    print(f"{'='*60}")

    syn_feat_all, syn_y = [], []
    vae.eval()

    for cls in sorted(counts.keys()):
        n_needed = majority_count - counts[cls]
        if n_needed <= 0:
            print(f"  Class {cls}: already majority, skip")
            continue
        print(f"  Class {cls}: generating {n_needed} synthetic feature vectors")

        gen = []
        rem = n_needed
        while rem > 0:
            bs = min(rem, 256)
            with torch.no_grad():
                s = vae.generate(cls, num_samples=bs, device=device)
                if cfg.GEN_NOISE > 0:
                    s = s + torch.randn_like(s) * cfg.GEN_NOISE
                gen.append(s.cpu().numpy())
            rem -= bs

        gen = np.concatenate(gen, axis=0)[:n_needed]
        syn_feat_all.append(gen)
        syn_y.extend([cls] * n_needed)

    if not syn_feat_all:
        return np.empty((0, 15)), np.array([])

    syn_feat = np.concatenate(syn_feat_all, axis=0)
    syn_y    = np.array(syn_y)
    print(f"  Total synthetic: {len(syn_y)}, shape: {syn_feat.shape}")
    return syn_feat, syn_y


# ========================== PHASE 3: CLASSIFIER ============================

def train_classifier(X_feat_tr, y_tr, X_feat_te, y_te, tag, cfg):
    device    = cfg.DEVICE
    clf       = create_classifier(X_feat_tr.shape[-1]).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=cfg.CLF_LR,
                                  weight_decay=cfg.CLF_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.CLF_EPOCHS)
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 预加载到 device
    X_tr_dev = torch.FloatTensor(X_feat_tr).to(device)
    y_tr_dev = torch.FloatTensor(y_tr).to(device)
    X_te_dev = torch.FloatTensor(X_feat_te).to(device)
    y_te_dev = torch.FloatTensor(y_te).to(device)

    train_dl = DataLoader(TensorDataset(X_tr_dev, y_tr_dev),
                          batch_size=cfg.CLF_BATCH, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te_dev, y_te_dev),
                          batch_size=cfg.CLF_BATCH)

    history  = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1, patience = 0, 0

    for epoch in range(cfg.CLF_EPOCHS):
        clf.train()
        total_loss, n = 0, 0
        for xb, yb in train_dl:
            loss = criterion(clf(xb).squeeze(-1), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(yb); n += len(yb)
        scheduler.step()

        clf.eval()
        ap, at, vl, vn = [], [], 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                logits = clf(xb).squeeze(-1)
                vl += criterion(logits, yb).item() * len(yb); vn += len(yb)
                ap.extend((torch.sigmoid(logits) > 0.35).float().cpu().tolist())
                at.extend(yb.cpu().tolist())

        ap, at = np.array(ap), np.array(at)
        acc    = (ap == at).mean()
        tp     = ((ap == 1) & (at == 1)).sum()
        fp     = ((ap == 1) & (at == 0)).sum()
        fn     = ((ap == 0) & (at == 1)).sum()
        prec   = tp / (tp + fp + 1e-8)
        rec    = tp / (tp + fn + 1e-8)
        f1     = 2 * prec * rec / (prec + rec + 1e-8)

        history['train_loss'].append(total_loss / n)
        history['val_loss'].append(vl / vn)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)

        if f1 > best_f1:
            best_f1 = f1; patience = 0
            torch.save(clf.state_dict(),
                       os.path.join(cfg.CKPT_DIR, f'clf_best_{tag}.pt'))
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print(f"    [{tag}] Early stop at epoch {epoch+1}"); break

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"    [{tag}] Epoch {epoch+1:4d} | "
                  f"Loss: {total_loss/n:.4f} | Acc: {acc:.3f} | F1: {f1:.3f}")

    clf.load_state_dict(torch.load(
        os.path.join(cfg.CKPT_DIR, f'clf_best_{tag}.pt'),
        map_location=device, weights_only=True))
    print(f"    [{tag}] Best F1: {best_f1:.3f}")
    return clf, history, best_f1


def train_classifiers(X_feat_tr, y_tr, X_feat_te, y_te, syn_feat, syn_y, cfg):
    """
    syn_feat 是 Feature-VAE 直接生成的归一化特征，无需再归一化，直接拼接。
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: Training Classifier on Real + Synthetic Features")
    print(f"  Real: {len(y_tr)}, Synthetic: {len(syn_y)}")
    print(f"{'='*60}")

    X_aug = np.concatenate([X_feat_tr, syn_feat], axis=0)
    y_aug = np.concatenate([y_tr,      syn_y],    axis=0)
    print(f"  Augmented total: {len(y_aug)} "
          f"({dict(Counter(y_aug.tolist()))})")

    clf_aug, hist_aug, f1_aug = train_classifier(
        X_aug, y_aug, X_feat_te, y_te, 'aug', cfg)

    print(f"\n  Final F1 (After VAE): {f1_aug:.3f}")
    return {'aug': hist_aug}


# ========================== MAIN ===========================================

def main():
    cfg = TrainConfig()
    set_seed(cfg.SEED)
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    split, raw_mean, raw_std, feat_mean, feat_std, vae_names, clin_names = \
        load_data(cfg)

    # Phase 1: LSTM-VAE 在特征空间训练
    vae, vae_hist = train_vae(split['X_feat_tr'], split['y_tr'], cfg)

    # Phase 2: 直接生成归一化特征（跳过原始信号生成）
    syn_feat, syn_y = generate_synthetic(vae, split['y_tr'], cfg)

    # Phase 3: Classifier
    clf_hists = train_classifiers(
        split['X_feat_tr'], split['y_tr'],
        split['X_feat_te'], split['y_te'],
        syn_feat, syn_y, cfg)

    # ── Loss 趋势图 ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ex = range(1, len(vae_hist['recon']) + 1)
    ax.plot(ex, vae_hist['total'], 'b-', lw=2, label='Total Loss')
    ax.plot(ex, vae_hist['recon'], 'g-', lw=2, label='Recon Loss')
    ax.axvline(x=cfg.VAE_WARMUP_EPOCHS, color='gray', ls=':', lw=1.5,
               label=f'Warmup end (ep {cfg.VAE_WARMUP_EPOCHS})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('LSTM Feature-VAE — Loss Trend')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'vae_loss_trend.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)

    h  = clf_hists['aug']
    ex = range(1, len(h['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ex, h['train_loss'], 'b-',  lw=2, label='Train Loss')
    ax.plot(ex, h['val_loss'],   'r--', lw=2, label='Val Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Classifier (After VAE) — Loss Trend')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'clf_loss_trend.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {results_dir}/vae_loss_trend.png, clf_loss_trend.png")

    # ── 保存结果 ──────────────────────────────────────────────────────────
    save_dict = {
        'vae_history':   vae_hist,
        'clf_histories': clf_hists,
        'best_model':    'aug',
        'feat_mean':     feat_mean,
        'feat_std':      feat_std,
        'raw_mean':      raw_mean,
        'raw_std':       raw_std,
        'vae_channel_names':      vae_names,
        'clinical_feature_names': clin_names,
        'X_raw_test':   split['X_raw_te'],
        'X_feat_test':  split['X_feat_te'],
        'y_test':       split['y_te'],
        'X_raw_train':  split['X_raw_tr'],
        'X_feat_train': split['X_feat_tr'],
        'y_train':      split['y_tr'],
        'syn_feat':     syn_feat,
        'syn_y':        syn_y,
        'file_ids_test':  split.get('fid_te'),
        'file_ids_train': split.get('fid_tr'),
    }
    with open(os.path.join(cfg.CKPT_DIR, 'train_results.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")


if __name__ == "__main__":
    main()