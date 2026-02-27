import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from model import create_vae, create_classifier, vae_loss_fn
from preprocessing import (
    patient_aware_split, compute_normalization_stats, normalize,
    extract_features_batch
)


class TrainConfig:
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_PATH = 'preprocessed/preprocessed_data.pkl'
    CKPT_DIR = 'checkpoints'

    # VAE training — two phases
    VAE_WARMUP_EPOCHS = 100   # Phase 1a: β=0, pure reconstruction
    VAE_ANNEAL_EPOCHS = 200   # Phase 1b: β ramps 0 → KL_MAX
    VAE_BATCH = 64
    VAE_LR = 1e-3
    KL_MAX = 0.005            # Very low max KL weight (was 0.1 → caused collapse)
    FREE_BITS = 0.1

    # Classifier
    CLF_EPOCHS = 200
    CLF_BATCH = 32
    CLF_LR = 1e-3
    CLF_WD = 1e-2
    LABEL_SMOOTH = 0.1

    GEN_NOISE = 0.05
    PATIENCE = 40


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
    pids = data['patient_ids']
    fids = data.get('file_ids', None)
    vae_names = data.get('vae_channel_names', [])
    clin_names = data['clinical_feature_names']

    print(f"[Load] X_raw={X_raw.shape}, X_feat={X_feat.shape}, "
          f"y: {dict(Counter(y))}, patients={len(set(pids))}")

    split = patient_aware_split(X_raw, X_feat, y, pids, fids)

    print(f"[Split] Train: {len(split['y_tr'])} ({dict(Counter(split['y_tr']))})")
    print(f"[Split] Test:  {len(split['y_te'])} ({dict(Counter(split['y_te']))})")

    # Normalize raw 6-ch
    raw_mean, raw_std = compute_normalization_stats(split['X_raw_tr'])
    split['X_raw_tr'] = normalize(split['X_raw_tr'], raw_mean, raw_std)
    split['X_raw_te'] = normalize(split['X_raw_te'], raw_mean, raw_std)

    # Normalize 15 features
    feat_mean = split['X_feat_tr'].mean(axis=0)
    feat_std = split['X_feat_tr'].std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    split['X_feat_tr'] = (split['X_feat_tr'] - feat_mean) / feat_std
    split['X_feat_te'] = (split['X_feat_te'] - feat_mean) / feat_std

    return split, raw_mean, raw_std, feat_mean, feat_std, vae_names, clin_names


# ========================== PHASE 1: VAE ===================================

def train_vae(X_raw_tr, y_tr, cfg):
    device = cfg.DEVICE
    input_dim = X_raw_tr.shape[-1]
    seq_len = X_raw_tr.shape[1]
    total_epochs = cfg.VAE_WARMUP_EPOCHS + cfg.VAE_ANNEAL_EPOCHS

    vae = create_vae(input_dim, seq_len).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_raw_tr), torch.LongTensor(y_tr)),
        batch_size=cfg.VAE_BATCH, shuffle=True)

    print(f"\n{'='*60}")
    print(f"PHASE 1: Training LSTM-VAE on {input_dim}-channel time series")
    print(f"  Phase 1a: {cfg.VAE_WARMUP_EPOCHS} epochs WARMUP (β=0, pure recon)")
    print(f"  Phase 1b: {cfg.VAE_ANNEAL_EPOCHS} epochs ANNEAL (β: 0 → {cfg.KL_MAX})")
    print(f"  Total: {total_epochs} epochs, batch={cfg.VAE_BATCH}")
    print(f"{'='*60}")

    history = {'total': [], 'recon': [], 'kl': [], 'kl_weight': []}
    best_recon = float('inf')

    for epoch in range(total_epochs):
        # Compute β (KL weight)
        if epoch < cfg.VAE_WARMUP_EPOCHS:
            # Phase 1a: pure reconstruction, no KL
            kl_w = 0.0
            phase = "warmup"
        else:
            # Phase 1b: linear ramp from 0 to KL_MAX
            progress = (epoch - cfg.VAE_WARMUP_EPOCHS) / cfg.VAE_ANNEAL_EPOCHS
            kl_w = cfg.KL_MAX * progress
            phase = "anneal"

        vae.train()
        ep_loss, ep_recon, ep_kl, n = 0, 0, 0, 0
        t0 = time.time()

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, lv = vae(xb, yb)
            loss, rl, kl = vae_loss_fn(recon, xb, mu, lv,
                                        kl_weight=kl_w, free_bits=cfg.FREE_BITS)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            bs = xb.size(0)
            ep_loss += loss.item()*bs; ep_recon += rl.item()*bs
            ep_kl += kl.item()*bs; n += bs

        scheduler.step()
        avg_recon = ep_recon / n
        history['total'].append(ep_loss/n)
        history['recon'].append(avg_recon)
        history['kl'].append(ep_kl/n)
        history['kl_weight'].append(kl_w)

        if (epoch+1) % 25 == 0 or epoch == 0:
            print(f"  [{phase:6s}] Epoch {epoch+1:3d}/{total_epochs} | "
                  f"Recon: {avg_recon:.4f} | KL: {ep_kl/n:.2f} | "
                  f"β: {kl_w:.5f} | {time.time()-t0:.1f}s")

        # Save best model (based on recon loss)
        if avg_recon < best_recon:
            best_recon = avg_recon
            ckpt_path = os.path.join(cfg.CKPT_DIR, 'vae_best.pt')
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            torch.save(vae.state_dict(), ckpt_path)

    vae.load_state_dict(torch.load(os.path.join(cfg.CKPT_DIR, 'vae_best.pt'),
                                    map_location=device, weights_only=True))

    # Quality check — per-channel correlation
    vae.eval()
    with torch.no_grad():
        n_check = min(64, len(X_raw_tr))
        xb = torch.FloatTensor(X_raw_tr[:n_check]).to(device)
        yb = torch.LongTensor(y_tr[:n_check]).to(device)
        recon, _, _ = vae(xb, yb)
        orig = xb.cpu().numpy()
        rec = recon.cpu().numpy()
        corrs_per_ch = {ch: [] for ch in range(input_dim)}
        for i in range(n_check):
            for ch in range(input_dim):
                r = np.corrcoef(orig[i, :, ch], rec[i, :, ch])[0, 1]
                if not np.isnan(r):
                    corrs_per_ch[ch].append(r)

        print(f"\n  [VAE Quality Check]")
        all_corrs = []
        for ch in range(input_dim):
            mc = np.mean(corrs_per_ch[ch])
            all_corrs.extend(corrs_per_ch[ch])
            print(f"    ch{ch}: mean_r={mc:.3f}")
        print(f"    Overall: mean_r={np.mean(all_corrs):.3f}, "
              f"median_r={np.median(all_corrs):.3f}")

    return vae, history


# ========================== PHASE 2: GENERATE ==============================

def generate_synthetic(vae, y_tr, raw_mean, raw_std, cfg):
    device = cfg.DEVICE
    counts = Counter(y_tr)
    majority_count = max(counts.values())

    print(f"\n{'='*60}")
    print(f"PHASE 2: Generating Synthetic Data (balance to {majority_count})")
    print(f"  Class counts: {dict(counts)}")
    print(f"{'='*60}")

    syn_raw_all, syn_y = [], []
    vae.eval()

    for cls in sorted(counts.keys()):
        n_needed = majority_count - counts[cls]
        if n_needed <= 0:
            print(f"  Class {cls}: already majority, skip")
            continue
        print(f"  Class {cls}: minority, generating {n_needed} to balance")
        gen = []
        rem = n_needed
        while rem > 0:
            bs = min(rem, 128)
            with torch.no_grad():
                s = vae.generate(cls, num_samples=bs, device=device)
                if cfg.GEN_NOISE > 0:
                    s += torch.randn_like(s) * cfg.GEN_NOISE
                gen.append(s.cpu().numpy())
            rem -= bs
        gen = np.concatenate(gen, axis=0)[:n_needed]
        syn_raw_all.append(gen)
        syn_y.extend([cls] * n_needed)

    if not syn_raw_all:
        return np.empty((0, 300, 6)), np.empty((0, 15)), np.array([])

    syn_raw = np.concatenate(syn_raw_all, axis=0)
    syn_y = np.array(syn_y)

    syn_denorm = syn_raw * raw_std + raw_mean
    syn_feat = extract_features_batch(syn_denorm, dt=1.0/60)

    print(f"  Total synthetic: {len(syn_y)}, feat shape: {syn_feat.shape}")
    return syn_raw, syn_feat, syn_y


# ========================== PHASE 3: CLASSIFIER ============================

def train_classifier(X_feat_tr, y_tr, X_feat_te, y_te, tag, cfg):
    device = cfg.DEVICE
    clf = create_classifier(X_feat_tr.shape[-1]).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=cfg.CLF_LR,
                                  weight_decay=cfg.CLF_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.CLF_EPOCHS)
    criterion = LabelSmoothingBCELoss(cfg.LABEL_SMOOTH)

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_feat_tr), torch.FloatTensor(y_tr)),
        batch_size=cfg.CLF_BATCH, shuffle=True)
    test_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_feat_te), torch.FloatTensor(y_te)),
        batch_size=cfg.CLF_BATCH)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1, patience = 0, 0

    for epoch in range(cfg.CLF_EPOCHS):
        clf.train()
        total_loss, n = 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(clf(xb).squeeze(-1), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()*len(yb); n += len(yb)
        scheduler.step()

        clf.eval()
        ap, at, vl, vn = [], [], 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = clf(xb).squeeze(-1)
                vl += criterion(logits, yb).item()*len(yb); vn += len(yb)
                ap.extend((torch.sigmoid(logits) > 0.5).float().cpu().tolist())
                at.extend(yb.cpu().tolist())

        ap, at = np.array(ap), np.array(at)
        acc = (ap == at).mean()
        tp = ((ap==1)&(at==1)).sum()
        fp = ((ap==1)&(at==0)).sum()
        fn = ((ap==0)&(at==1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        history['train_loss'].append(total_loss/n)
        history['val_loss'].append(vl/vn)
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

        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"    [{tag}] Epoch {epoch+1:3d} | "
                  f"Loss: {total_loss/n:.4f} | Acc: {acc:.3f} | F1: {f1:.3f}")

    clf.load_state_dict(torch.load(
        os.path.join(cfg.CKPT_DIR, f'clf_best_{tag}.pt'),
        map_location=device, weights_only=True))
    print(f"    [{tag}] Best F1: {best_f1:.3f}")
    return clf, history, best_f1


def train_classifiers(X_feat_tr, y_tr, X_feat_te, y_te,
                      syn_feat, syn_y, feat_mean, feat_std, cfg):
    print(f"\n{'='*60}")
    print(f"PHASE 3: Training Classifier on Real + Synthetic Features")
    print(f"  Real: {len(y_tr)}, Synthetic: {len(syn_y)}")
    print(f"{'='*60}")

    syn_feat_norm = (syn_feat - feat_mean) / feat_std
    X_aug = np.concatenate([X_feat_tr, syn_feat_norm], axis=0)
    y_aug = np.concatenate([y_tr, syn_y], axis=0)
    print(f"\n  --- Augmented (real+syn={len(y_aug)}) ---")
    clf_aug, hist_aug, f1_aug = train_classifier(
        X_aug, y_aug, X_feat_te, y_te, 'aug', cfg)

    print(f"\n  Final F1 (After VAE): {f1_aug:.3f}")
    return {'aug': hist_aug}


# ========================== MAIN ===========================================

def main():
    cfg = TrainConfig()
    set_seed(cfg.SEED)
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)

    split, raw_mean, raw_std, feat_mean, feat_std, vae_names, clin_names = \
        load_data(cfg)

    vae, vae_hist = train_vae(split['X_raw_tr'], split['y_tr'], cfg)
    syn_raw, syn_feat, syn_y = generate_synthetic(
        vae, split['y_tr'], raw_mean, raw_std, cfg)
    clf_hists = train_classifiers(
        split['X_feat_tr'], split['y_tr'],
        split['X_feat_te'], split['y_te'],
        syn_feat, syn_y, feat_mean, feat_std, cfg)

    save_dict = {
        'vae_history': vae_hist, 'clf_histories': clf_hists,
        'best_model': 'aug',
        'raw_mean': raw_mean, 'raw_std': raw_std,
        'feat_mean': feat_mean, 'feat_std': feat_std,
        'vae_channel_names': vae_names,
        'clinical_feature_names': clin_names,
        'X_raw_test': split['X_raw_te'], 'X_feat_test': split['X_feat_te'],
        'y_test': split['y_te'],
        'X_raw_train': split['X_raw_tr'], 'X_feat_train': split['X_feat_tr'],
        'y_train': split['y_tr'],
        'syn_raw': syn_raw, 'syn_feat': syn_feat, 'syn_y': syn_y,
        'file_ids_test': split.get('fid_te'),
        'file_ids_train': split.get('fid_tr'),
    }
    with open(os.path.join(cfg.CKPT_DIR, 'train_results.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"\n{'='*60}\nTRAINING COMPLETE\n{'='*60}")


if __name__ == "__main__":
    main()