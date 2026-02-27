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

    # VAE training — two phases (extended for better reconstruction)
    VAE_WARMUP_EPOCHS = 150   # Phase 1a: β=0, pure reconstruction (was 100)
    VAE_ANNEAL_EPOCHS = 250   # Phase 1b: β ramps 0 → KL_MAX (was 200)
    VAE_BATCH = 64
    VAE_LR = 1e-3
    KL_MAX = 0.005
    FREE_BITS = 0.1

    # Classifier
    CLF_EPOCHS = 300          # was 200
    CLF_BATCH = 32
    CLF_LR = 1e-3
    CLF_WD = 1e-2
    LABEL_SMOOTH = 0.05       # was 0.1, reduced for cleaner signal
    GEN_NOISE = 0.02          # was 0.05, less noise for cleaner synthesis
    PATIENCE = 50             # was 40

    # ── Strategy 1 only — improved posterior interpolation ───────────────
    OVERSAMPLE_FACTOR    = 1.5   # generate 1.5× gap to provide more data
    KNN_NEIGHBORS        = 5     # mix anchor with one of its k nearest latent neighbours
    BETA_ALPHA           = 2.0   # Beta(α,α): α=2 concentrates near 0.5 (real mixing)
    QUALITY_FILTER_RATIO = 2     # generate 2× needed, keep best by proximity to real


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, s=0.05):
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
    feat_std  = split['X_feat_tr'].std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    split['X_feat_tr'] = (split['X_feat_tr'] - feat_mean) / feat_std
    split['X_feat_te'] = (split['X_feat_te'] - feat_mean) / feat_std

    return split, raw_mean, raw_std, feat_mean, feat_std, vae_names, clin_names


# ========================== PHASE 1: VAE ===================================

def train_vae(X_raw_tr, y_tr, cfg):
    device = cfg.DEVICE
    input_dim = X_raw_tr.shape[-1]
    seq_len   = X_raw_tr.shape[1]
    total_epochs = cfg.VAE_WARMUP_EPOCHS + cfg.VAE_ANNEAL_EPOCHS

    vae = create_vae(input_dim, seq_len).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.VAE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_raw_tr), torch.LongTensor(y_tr)),
        batch_size=cfg.VAE_BATCH, shuffle=True)

    print(f"\n{'='*60}")
    print(f"PHASE 1: Training Conditional LSTM-VAE")
    print(f"  Phase 1a: {cfg.VAE_WARMUP_EPOCHS} epochs  WARMUP  (β=0, pure recon)")
    print(f"  Phase 1b: {cfg.VAE_ANNEAL_EPOCHS} epochs  ANNEAL  (β: 0 → {cfg.KL_MAX})")
    print(f"  Total: {total_epochs} epochs, batch={cfg.VAE_BATCH}, device={device}")
    print(f"{'='*60}")

    history = {'total': [], 'recon': [], 'kl': [], 'kl_weight': []}
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
            ep_kl   += kl.item()*bs;  n += bs

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

        if avg_recon < best_recon:
            best_recon = avg_recon
            torch.save(vae.state_dict(),
                       os.path.join(cfg.CKPT_DIR, 'vae_best.pt'))

    vae.load_state_dict(torch.load(os.path.join(cfg.CKPT_DIR, 'vae_best.pt'),
                                    map_location=device, weights_only=True))

    # ── Quality check: per-channel reconstruction correlation ──
    vae.eval()
    with torch.no_grad():
        n_check = min(64, len(X_raw_tr))
        xb = torch.FloatTensor(X_raw_tr[:n_check]).to(device)
        yb = torch.LongTensor(y_tr[:n_check]).to(device)
        recon, _, _ = vae(xb, yb)
        orig = xb.cpu().numpy()
        rec  = recon.cpu().numpy()
        corrs_per_ch = {ch: [] for ch in range(input_dim)}
        for i in range(n_check):
            for ch in range(input_dim):
                r = np.corrcoef(orig[i, :, ch], rec[i, :, ch])[0, 1]
                if not np.isnan(r):
                    corrs_per_ch[ch].append(r)

        print(f"\n  [VAE Quality Check — per-channel mean_r]")
        all_corrs = []
        for ch in range(input_dim):
            mc = np.mean(corrs_per_ch[ch])
            all_corrs.extend(corrs_per_ch[ch])
            print(f"    ch{ch}: mean_r={mc:.3f}")
        print(f"    Overall: mean_r={np.mean(all_corrs):.3f}, "
              f"median_r={np.median(all_corrs):.3f}")

    return vae, history


# ========================== PHASE 2a: POSTERIOR GENERATION =================

def generate_synthetic_posterior(vae, X_raw_tr, y_tr, raw_mean, raw_std, cfg):
    """
    Improved Posterior Interpolation Generation (Strategy 1).

    Three targeted enhancements over the naive posterior approach:

    1. KNN-guided neighbour selection
       Encode all real samples → mu space.  For each new sample, pick an
       anchor i at random, then pick partner j from i's k nearest neighbours
       (not globally random).  Interpolation stays within a local,
       geometrically consistent neighbourhood on the data manifold.

    2. Beta(α, α) mixing weight  (α = cfg.BETA_ALPHA, default 2.0)
       alpha ~ Beta(2,2) concentrates near 0.5, so both parents always
       contribute meaningfully.  Uniform(0,1) can produce near-copies of
       a single sample; Beta(2,2) prevents that.

    3. Quality filtering via re-encoding
       Generate QUALITY_FILTER_RATIO × n_needed candidates, re-encode each
       synthetic sequence through the VAE, and keep only the n_needed whose
       latent code is closest to any real sample (1-NN distance).  This
       discards off-manifold outliers before they enter training.
    """
    from sklearn.neighbors import NearestNeighbors as SklearnNN

    device = cfg.DEVICE
    counts = Counter(y_tr.tolist())
    majority_count = max(counts.values())

    print(f"\n{'='*60}")
    print(f"PHASE 2b: Improved Posterior Interpolation (Strategy 1)")
    print(f"  KNN neighbours : {cfg.KNN_NEIGHBORS}")
    print(f"  Beta alpha     : {cfg.BETA_ALPHA}  (Beta({cfg.BETA_ALPHA},{cfg.BETA_ALPHA}))")
    print(f"  Filter ratio   : {cfg.QUALITY_FILTER_RATIO}× → keep best")
    print(f"  Oversample     : {cfg.OVERSAMPLE_FACTOR}")
    print(f"  Class counts   : {dict(counts)}")
    print(f"{'='*60}")

    syn_raw_all, syn_y = [], []
    vae.eval()

    for cls in sorted(counts.keys()):
        n_balance = majority_count - counts[cls]
        if n_balance <= 0:
            print(f"  Class {cls}: already majority — skip")
            continue

        n_needed   = max(1, int(n_balance * cfg.OVERSAMPLE_FACTOR))
        n_generate = n_needed * cfg.QUALITY_FILTER_RATIO
        print(f"  Class {cls}: minority by {n_balance}, "
              f"generating {n_generate} → keeping {n_needed}")

        # ── Step 1: encode all real samples ─────────────────────────────
        mask  = (y_tr == cls)
        X_cls = X_raw_tr[mask]

        mu_list, logvar_list = [], []
        for i in range(0, len(X_cls), 64):
            bend = min(i + 64, len(X_cls))
            xb = torch.FloatTensor(X_cls[i:bend]).to(device)
            yb = torch.LongTensor([cls] * (bend - i)).to(device)
            with torch.no_grad():
                _, mu, logvar = vae.encode(xb, yb)
            mu_list.append(mu.cpu())
            logvar_list.append(logvar.cpu())

        mu_all  = torch.cat(mu_list,   dim=0)          # (N_cls, latent_dim)
        std_all = torch.exp(0.5 * torch.cat(logvar_list, dim=0))
        mu_np   = mu_all.numpy()

        # ── Step 2: KNN in latent space ──────────────────────────────────
        k = min(cfg.KNN_NEIGHBORS, len(mu_all) - 1)
        nn_model = SklearnNN(n_neighbors=k + 1).fit(mu_np)
        _, knn_idx = nn_model.kneighbors(mu_np)        # (N_cls, k+1)
        knn_idx = knn_idx[:, 1:]                        # exclude self → (N_cls, k)

        # ── Step 3: generate n_generate candidates ───────────────────────
        gen = []
        rem = n_generate
        while rem > 0:
            bs = min(rem, 128)

            # Anchor i random; partner j from i's KNN
            i_arr = np.random.randint(0, len(mu_all), bs)
            j_arr = np.array([knn_idx[i][np.random.randint(0, k)]
                              for i in i_arr])

            # Beta(α,α) mixing weight
            alpha = torch.tensor(
                np.random.beta(cfg.BETA_ALPHA, cfg.BETA_ALPHA, (bs, 1)),
                dtype=torch.float32)

            i_t = torch.tensor(i_arr)
            j_t = torch.tensor(j_arr)
            mu_mix  = alpha * mu_all[i_t]  + (1 - alpha) * mu_all[j_t]
            std_mix = alpha * std_all[i_t] + (1 - alpha) * std_all[j_t]

            z      = mu_mix + std_mix * torch.randn_like(std_mix)
            z      = z.to(device)
            labels = torch.full((bs,), cls, dtype=torch.long).to(device)

            with torch.no_grad():
                s = vae.decode(z, labels)
                if cfg.GEN_NOISE > 0:
                    s = s + torch.randn_like(s) * cfg.GEN_NOISE

            gen.append(s.cpu().numpy())
            rem -= bs

        gen_arr = np.concatenate(gen, axis=0)[:n_generate]

        # ── Step 4: quality filter — keep closest to any real sample ─────
        if cfg.QUALITY_FILTER_RATIO > 1 and len(gen_arr) > n_needed:
            # Re-encode synthetic sequences
            syn_mu_list = []
            for i in range(0, len(gen_arr), 64):
                bend = min(i + 64, len(gen_arr))
                xb = torch.FloatTensor(gen_arr[i:bend]).to(device)
                yb = torch.LongTensor([cls] * (bend - i)).to(device)
                with torch.no_grad():
                    _, mu_syn, _ = vae.encode(xb, yb)
                syn_mu_list.append(mu_syn.cpu().numpy())

            syn_mu = np.concatenate(syn_mu_list, axis=0)

            # 1-NN distance to real latent codes
            nn_filter = SklearnNN(n_neighbors=1).fit(mu_np)
            dists, _  = nn_filter.kneighbors(syn_mu)
            min_dists  = dists[:, 0]

            keep_idx = np.argsort(min_dists)[:n_needed]
            gen_arr  = gen_arr[keep_idx]
            print(f"    Quality filter: "
                  f"avg d(nearest real) kept={min_dists[keep_idx].mean():.3f} "
                  f"vs all={min_dists.mean():.3f}")
        else:
            gen_arr = gen_arr[:n_needed]

        syn_raw_all.append(gen_arr)
        syn_y.extend([cls] * len(gen_arr))

    if not syn_raw_all:
        return np.empty((0, 300, 6)), np.empty((0, 15)), np.array([], dtype=np.int64)

    syn_raw = np.concatenate(syn_raw_all, axis=0)
    syn_y   = np.array(syn_y, dtype=np.int64)

    syn_denorm = syn_raw * raw_std + raw_mean
    syn_feat   = extract_features_batch(syn_denorm, dt=1.0 / 60)

    print(f"  Strategy 1 generated: {len(syn_y)} samples | feat={syn_feat.shape}")
    print(f"  Distribution: "
          f"cls0={counts.get(0,0)+int((syn_y==0).sum())}, "
          f"cls1={counts.get(1,0)+int((syn_y==1).sum())}")
    return syn_raw, syn_feat, syn_y


def generate_synthetic_prior(vae, y_tr, raw_mean, raw_std, cfg):
    """
    Fallback: original prior sampling z ~ N(0,I).
    Kept for ablation comparison.
    """
    device = cfg.DEVICE
    counts = Counter(y_tr.tolist())
    majority_count = max(counts.values())

    print(f"\n{'='*60}")
    print(f"PHASE 2a: Prior Sampling Generation (fallback)")
    print(f"{'='*60}")

    syn_raw_all, syn_y = [], []
    vae.eval()

    for cls in sorted(counts.keys()):
        n_needed = majority_count - counts[cls]
        if n_needed <= 0:
            print(f"  Class {cls}: majority, skip")
            continue
        print(f"  Class {cls}: generating {n_needed}")
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
        return np.empty((0, 300, 6)), np.empty((0, 15)), np.array([], dtype=np.int64)

    syn_raw    = np.concatenate(syn_raw_all, axis=0)
    syn_y      = np.array(syn_y, dtype=np.int64)
    syn_denorm = syn_raw * raw_std + raw_mean
    syn_feat   = extract_features_batch(syn_denorm, dt=1.0 / 60)

    print(f"  Total synthetic: {len(syn_y)}, feat shape: {syn_feat.shape}")
    return syn_raw, syn_feat, syn_y


# ========================== PHASE 2b: SMOTE ================================

def smote_on_features(X_feat, y, k=5):
    """
    Improvement 3 — SMOTE in 15-dimensional clinical feature space.

    For each minority sample, find k nearest neighbours and interpolate
    linearly between the sample and a random neighbour to create a new
    synthetic feature vector.  More reliable than VAE-based augmentation
    because it operates directly in the discriminative feature space.
    """
    from sklearn.neighbors import NearestNeighbors

    counts        = Counter(y.tolist())
    majority_count = max(counts.values())

    syn_feat_list, syn_y_list = [], []

    for cls, count in counts.items():
        n_needed = majority_count - count
        if n_needed <= 0:
            continue

        X_cls    = X_feat[y == cls]
        k_actual = min(k, len(X_cls) - 1)
        if k_actual < 1:
            continue

        nn_model = NearestNeighbors(n_neighbors=k_actual + 1).fit(X_cls)
        _, indices = nn_model.kneighbors(X_cls)

        syn = []
        for _ in range(n_needed):
            i      = np.random.randint(0, len(X_cls))
            nn_idx = indices[i][1:]          # exclude self
            j      = nn_idx[np.random.randint(0, len(nn_idx))]
            gap    = np.random.random()
            syn.append(X_cls[i] + gap * (X_cls[j] - X_cls[i]))

        syn_feat_list.append(np.array(syn, dtype=np.float32))
        syn_y_list.extend([cls] * n_needed)

    if not syn_feat_list:
        return (np.empty((0, X_feat.shape[1]), dtype=np.float32),
                np.array([], dtype=np.int64))

    return (np.concatenate(syn_feat_list, axis=0),
            np.array(syn_y_list, dtype=np.int64))


# ========================== PHASE 2c: LATENT EXTRACTION ====================

def extract_latent_features(vae, X_raw, y, device, batch_size=64):
    """
    Improvement 4 — Extract VAE encoder mean vectors (mu) as features.

    The encoder has learned a compact representation of the tapping
    sequences.  Concatenating these 64-dim latent features with the
    15 clinical features gives the combined (79-dim) classifier richer
    information than either alone.
    """
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


# ========================== PHASE 3: CLASSIFIER ============================

def train_classifier(X_feat_tr, y_tr, X_feat_te, y_te,
                     tag, cfg, pos_weight=None):
    """
    Train one MLP classifier variant.
    pos_weight: if given, use class-weighted BCE (Improvement 2).
                Computed as count(class0)/count(class1) for real-only.
    """
    device    = cfg.DEVICE
    input_dim = X_feat_tr.shape[-1]
    clf       = create_classifier(input_dim).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=cfg.CLF_LR,
                                  weight_decay=cfg.CLF_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.CLF_EPOCHS)

    # ── Improvement 2: class-weighted BCE for minority class ──
    if pos_weight is not None:
        pw        = torch.tensor([pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"    [{tag}] Class-weighted BCE  pos_weight={pos_weight:.3f}")
    else:
        criterion = LabelSmoothingBCELoss(cfg.LABEL_SMOOTH)

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_feat_tr), torch.FloatTensor(y_tr)),
        batch_size=cfg.CLF_BATCH, shuffle=True)
    test_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_feat_te), torch.FloatTensor(y_te)),
        batch_size=cfg.CLF_BATCH)

    history = {'train_loss': [], 'val_loss': [],
               'val_acc': [], 'val_f1': [], 'val_recall_pd': []}
    best_f1, patience = 0, 0

    for epoch in range(cfg.CLF_EPOCHS):
        clf.train()
        total_loss, n = 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(clf(xb).squeeze(-1), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(yb); n += len(yb)
        scheduler.step()

        clf.eval()
        ap, at, vl, vn = [], [], 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits  = clf(xb).squeeze(-1)
                vl += criterion(logits, yb).item() * len(yb); vn += len(yb)
                ap.extend((torch.sigmoid(logits) > 0.5).float().cpu().tolist())
                at.extend(yb.cpu().tolist())

        ap, at = np.array(ap), np.array(at)
        acc = (ap == at).mean()
        tp  = ((ap==1)&(at==1)).sum()
        fp  = ((ap==1)&(at==0)).sum()
        fn  = ((ap==0)&(at==1)).sum()
        prec       = tp / (tp + fp + 1e-8)
        rec        = tp / (tp + fn + 1e-8)
        f1         = 2 * prec * rec / (prec + rec + 1e-8)
        recall_pd  = rec

        history['train_loss'].append(total_loss / n)
        history['val_loss'].append(vl / vn)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        history['val_recall_pd'].append(recall_pd)

        if f1 > best_f1:
            best_f1 = f1; patience = 0
            torch.save(clf.state_dict(),
                       os.path.join(cfg.CKPT_DIR, f'clf_best_{tag}.pt'))
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print(f"    [{tag}] Early stop @ epoch {epoch+1}"); break

        if (epoch+1) % 25 == 0 or epoch == 0:
            print(f"    [{tag}] Epoch {epoch+1:3d} | "
                  f"Loss: {total_loss/n:.4f} | "
                  f"Acc: {acc:.3f} | F1: {f1:.3f} | "
                  f"Prec: {prec:.3f} | PD_Rec: {recall_pd:.3f}")

    clf.load_state_dict(torch.load(
        os.path.join(cfg.CKPT_DIR, f'clf_best_{tag}.pt'),
        map_location=device, weights_only=True))
    print(f"    [{tag}] *** Best F1: {best_f1:.4f} ***")
    return clf, history, float(best_f1)


# ========================== PHASE 3: TWO-VARIANT COMPARISON ================

def train_all_classifiers(split, syn_prior, syn_posterior,
                          feat_mean, feat_std, cfg):
    """
    Train two classifiers for a clean ablation:

    baseline   — real data + prior sampling (z ~ N(0,I)), original approach
    strategy1  — real data + improved posterior interpolation

    Both use identical LabelSmoothingBCELoss so the only difference is the
    augmentation source.
    """
    print(f"\n{'='*60}")
    print(f"PHASE 3: Baseline vs Strategy 1 Comparison")
    print(f"  Input: {split['X_feat_tr'].shape[1]} clinical features")
    print(f"{'='*60}")

    X_feat_tr = split['X_feat_tr']
    y_tr      = split['y_tr']
    X_feat_te = split['X_feat_te']
    y_te      = split['y_te']

    results = {}

    # ── Variant 1: Baseline — prior sampling augmentation ───────────────
    syn_raw_p, syn_feat_p, syn_y_p = syn_prior
    if len(syn_y_p) > 0:
        syn_feat_p_norm = (syn_feat_p - feat_mean) / feat_std
        X_base = np.concatenate([X_feat_tr, syn_feat_p_norm], axis=0)
        y_base = np.concatenate([y_tr,      syn_y_p          ], axis=0)
    else:
        X_base, y_base = X_feat_tr, y_tr

    base_cnt = dict(Counter(y_base.tolist()))
    print(f"\n  [1/2] Baseline (prior sampling)  N={len(y_base)}  dist={base_cnt}")
    _, h1, f1_1 = train_classifier(X_base, y_base, X_feat_te, y_te,
                                    'baseline', cfg, pos_weight=None)
    results['baseline'] = (f1_1, h1)

    # ── Variant 2: Strategy 1 — improved posterior interpolation ────────
    syn_raw_s1, syn_feat_s1, syn_y_s1 = syn_posterior
    if len(syn_y_s1) > 0:
        syn_feat_s1_norm = (syn_feat_s1 - feat_mean) / feat_std
        X_s1 = np.concatenate([X_feat_tr, syn_feat_s1_norm], axis=0)
        y_s1 = np.concatenate([y_tr,      syn_y_s1         ], axis=0)
    else:
        X_s1, y_s1 = X_feat_tr, y_tr

    s1_cnt = dict(Counter(y_s1.tolist()))
    print(f"\n  [2/2] Strategy 1 (improved posterior)  N={len(y_s1)}  dist={s1_cnt}")
    _, h2, f1_2 = train_classifier(X_s1, y_s1, X_feat_te, y_te,
                                    'strategy1', cfg, pos_weight=None)
    results['strategy1'] = (f1_2, h2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3 RESULTS:")
    print(f"  {'Variant':22s}  {'Best F1':>8s}")
    print(f"  {'-'*34}")
    best_f1_val = max(v[0] for v in results.values())
    for tag, (f1, _) in sorted(results.items(), key=lambda x: -x[1][0]):
        marker = " ← BEST" if f1 == best_f1_val else ""
        print(f"  {tag:22s}  {f1:8.4f}{marker}")

    d_f1 = results['strategy1'][0] - results['baseline'][0]
    print(f"\n  Strategy 1 vs Baseline:  ΔF1(PD) = {d_f1:+.4f}")
    print(f"{'='*60}")

    best_tag  = max(results.items(), key=lambda x: x[1][0])[0]
    clf_hists = {tag: h for tag, (_, h) in results.items()}
    return best_tag, clf_hists


# ========================== MAIN ===========================================

def main():
    cfg = TrainConfig()
    set_seed(cfg.SEED)
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)

    split, raw_mean, raw_std, feat_mean, feat_std, vae_names, clin_names = \
        load_data(cfg)

    # Phase 1: Train Conditional LSTM-VAE
    vae, vae_hist = train_vae(split['X_raw_tr'], split['y_tr'], cfg)

    # Phase 2a: Baseline — prior sampling  z ~ N(0,I)
    syn_raw_prior, syn_feat_prior, syn_y_prior = generate_synthetic_prior(
        vae, split['y_tr'], raw_mean, raw_std, cfg)

    # Phase 2b: Strategy 1 — improved posterior interpolation
    syn_raw_post, syn_feat_post, syn_y_post = generate_synthetic_posterior(
        vae, split['X_raw_tr'], split['y_tr'], raw_mean, raw_std, cfg)

    # Phase 3: Baseline vs Strategy 1
    best_tag, clf_hists = train_all_classifiers(
        split,
        syn_prior     = (syn_raw_prior, syn_feat_prior, syn_y_prior),
        syn_posterior = (syn_raw_post,  syn_feat_post,  syn_y_post),
        feat_mean=feat_mean, feat_std=feat_std, cfg=cfg)

    # Save everything (use strategy1 synthetic for t-SNE visualisation)
    save_dict = {
        'vae_history':            vae_hist,
        'clf_histories':          clf_hists,
        'best_model':             best_tag,
        'raw_mean':               raw_mean,
        'raw_std':                raw_std,
        'feat_mean':              feat_mean,
        'feat_std':               feat_std,
        'vae_channel_names':      vae_names,
        'clinical_feature_names': clin_names,
        'X_raw_test':             split['X_raw_te'],
        'X_feat_test':            split['X_feat_te'],
        'y_test':                 split['y_te'],
        'X_raw_train':            split['X_raw_tr'],
        'X_feat_train':           split['X_feat_tr'],
        'y_train':                split['y_tr'],
        'syn_raw':                syn_raw_post,   # strategy1 for t-SNE
        'syn_feat':               syn_feat_post,
        'syn_y':                  syn_y_post,
        'file_ids_test':          split.get('fid_te'),
        'file_ids_train':         split.get('fid_tr'),
        'combined_stats':         None,
    }
    with open(os.path.join(cfg.CKPT_DIR, 'train_results.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best model: {best_tag}")
    print(f"  Checkpoints saved to: {cfg.CKPT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
