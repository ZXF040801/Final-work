import os, pickle, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

from preprocessing import patient_aware_split
from model_MLP  import (create_vae  as mlp_create_vae,
                         create_classifier as mlp_create_clf,
                         vae_loss_fn as mlp_vae_loss)
from model_LSTM import (create_vae  as lstm_create_vae,
                         create_classifier as lstm_create_clf,
                         vae_loss_fn as lstm_vae_loss)

warnings.filterwarnings('ignore')


SEEDS       = [42, 123, 256, 512, 1024, 2048, 314, 999, 777, 100]
DATA_PATH   = 'preprocessed/preprocessed_data.pkl'
RESULTS_DIR = 'results'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD   = 0.35

VAE_WARMUP  = 100
VAE_ANNEAL  = 200
VAE_BATCH   = 64
VAE_LR      = 1e-3
KL_MAX      = 0.005
FREE_BITS   = 0.1
GEN_NOISE   = 0.02

CLF_EPOCHS  = 500
CLF_BATCH   = 32
CLF_LR      = 1e-3
CLF_WD      = 1e-2
PATIENCE    = 9999


MLP_EPOCHS  = 500
MLP_BATCH   = 32
MLP_LR      = 5e-4
MLP_WD      = 5e-2
MLP_DROPOUT = 0.6
MLP_SMOOTH  = 0.1


LSTM_EPOCHS  = 500
LSTM_BATCH   = 32
LSTM_LR      = 1e-3
LSTM_WD      = 1e-2
LSTM_DROPOUT = 0.4
LSTM_SMOOTH  = 0.1



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_metrics(preds, probs, truth):
    tp = int(((preds == 1) & (truth == 1)).sum())
    tn = int(((preds == 0) & (truth == 0)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    mcc_denom   = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc         = (tp*tn - fp*fn) / (mcc_denom + 1e-8)

    fpr, tpr, _ = roc_curve(truth, probs)
    auc_score   = sk_auc(fpr, tpr)

    return sensitivity, specificity, mcc, auc_score


class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, s=0.1):
        super().__init__()
        self.s = s

    def forward(self, logits, targets):
        targets = targets.float() * (1 - self.s) + 0.5 * self.s
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)



def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data


def prepare_split(data, seed):
    set_seed(seed)
    X_raw  = data['X_raw']
    X_feat = data['X_feat']
    y      = data['y']
    pids   = data['patient_ids']
    fids   = data.get('file_ids', None)

    split = patient_aware_split(X_raw, X_feat, y, pids, fids)

    # 归一化
    feat_mean = split['X_feat_tr'].mean(axis=0)
    feat_std  = split['X_feat_tr'].std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    split['X_feat_tr'] = (split['X_feat_tr'] - feat_mean) / feat_std
    split['X_feat_te'] = (split['X_feat_te'] - feat_mean) / feat_std

    return split



class StandaloneMLP(nn.Module):
    def __init__(self, input_dim, dropout=MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_standalone(model, X_tr, y_tr, X_te, y_te, epochs, batch, lr, wd, smooth):
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te).to(DEVICE)
    y_te_t = torch.FloatTensor(y_te).to(DEVICE)

    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=batch, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te_t, y_te_t),
                          batch_size=batch)

    criterion = LabelSmoothingBCELoss(smooth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1    = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        probs_list, true_list = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                probs_list.extend(torch.sigmoid(model(xb)).cpu().tolist())
                true_list.extend(yb.cpu().tolist())
        probs = np.array(probs_list)
        truth = np.array(true_list, dtype=int)
        preds = (probs > THRESHOLD).astype(int)
        tp = ((preds==1)&(truth==1)).sum()
        fp = ((preds==1)&(truth==0)).sum()
        fn = ((preds==0)&(truth==1)).sum()
        prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
        f1   = 2*prec*rec/(prec+rec+1e-8)

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    probs_list, true_list = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            probs_list.extend(torch.sigmoid(model(xb)).cpu().tolist())
            true_list.extend(yb.cpu().tolist())
    probs = np.array(probs_list)
    truth = np.array(true_list, dtype=int)
    preds = (probs > THRESHOLD).astype(int)
    return calc_metrics(preds, probs, truth)



def train_vae_fn(vae, X_feat_tr, y_tr, loss_fn):
    total_epochs = VAE_WARMUP + VAE_ANNEAL
    X_dev = torch.FloatTensor(X_feat_tr).to(DEVICE)
    y_dev = torch.LongTensor(y_tr).to(DEVICE)
    loader = DataLoader(TensorDataset(X_dev, y_dev),
                        batch_size=VAE_BATCH, shuffle=True)

    optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5)

    best_recon = float('inf')
    best_state = None

    for epoch in range(total_epochs):
        kl_w = 0.0 if epoch < VAE_WARMUP else \
               KL_MAX * (epoch - VAE_WARMUP) / VAE_ANNEAL

        vae.train()
        ep_recon, n = 0, 0
        for xb, yb in loader:
            recon, mu, lv = vae(xb, yb)
            loss, rl, _   = loss_fn(recon, xb, mu, lv,
                                     kl_weight=kl_w, free_bits=FREE_BITS)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            ep_recon += rl.item() * xb.size(0)
            n        += xb.size(0)
        scheduler.step()

        avg_recon = ep_recon / n
        if avg_recon < best_recon:
            best_recon = avg_recon
            best_state = {k: v.clone() for k, v in vae.state_dict().items()}

    vae.load_state_dict(best_state)
    return vae


def generate_synthetic_fn(vae, y_tr):
    counts         = Counter(y_tr.tolist())
    majority_count = max(counts.values())
    syn_feats, syn_ys = [], []

    vae.eval()
    for cls in sorted(counts.keys()):
        n_needed = majority_count - counts[cls]
        if n_needed <= 0:
            continue
        gen, rem = [], n_needed
        while rem > 0:
            bs = min(rem, 256)
            with torch.no_grad():
                s = vae.generate(cls, num_samples=bs, device=DEVICE)
                if GEN_NOISE > 0:
                    s = s + torch.randn_like(s) * GEN_NOISE
                gen.append(s.cpu().numpy())
            rem -= bs
        syn_feats.append(np.concatenate(gen)[:n_needed])
        syn_ys.extend([cls] * n_needed)

    if not syn_feats:
        feat_dim = 15
        return np.empty((0, feat_dim)), np.array([])
    return np.concatenate(syn_feats), np.array(syn_ys)


def run_vae_model(create_vae_fn, create_clf_fn, loss_fn,
                  X_feat_tr, y_tr, X_feat_te, y_te):

    feat_dim = X_feat_tr.shape[-1]

    vae = create_vae_fn(feat_dim).to(DEVICE)
    vae = train_vae_fn(vae, X_feat_tr, y_tr, loss_fn)

    syn_feat, syn_y = generate_synthetic_fn(vae, y_tr)

    if len(syn_y) > 0:
        X_aug = np.concatenate([X_feat_tr, syn_feat])
        y_aug = np.concatenate([y_tr,      syn_y])
    else:
        X_aug, y_aug = X_feat_tr, y_tr

    clf = create_clf_fn(feat_dim).to(DEVICE)

    X_tr_t = torch.FloatTensor(X_aug).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_aug).to(DEVICE)
    X_te_t = torch.FloatTensor(X_feat_te).to(DEVICE)
    y_te_t = torch.FloatTensor(y_te).to(DEVICE)

    n_neg = (y_aug == 0).sum()
    n_pos = (y_aug == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=CLF_BATCH, shuffle=True)
    test_dl  = DataLoader(TensorDataset(X_te_t, y_te_t),
                          batch_size=CLF_BATCH)

    optimizer = torch.optim.Adam(clf.parameters(), lr=CLF_LR, weight_decay=CLF_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLF_EPOCHS)

    best_f1    = 0.0
    best_state = None

    for epoch in range(CLF_EPOCHS):
        clf.train()
        for xb, yb in train_dl:
            loss = criterion(clf(xb).squeeze(-1), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        clf.eval()
        probs_list, true_list = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                probs_list.extend(
                    torch.sigmoid(clf(xb).squeeze(-1)).cpu().tolist())
                true_list.extend(yb.cpu().tolist())
        probs = np.array(probs_list)
        truth = np.array(true_list, dtype=int)
        preds = (probs > THRESHOLD).astype(int)
        tp = ((preds==1)&(truth==1)).sum()
        fp = ((preds==1)&(truth==0)).sum()
        fn = ((preds==0)&(truth==1)).sum()
        prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
        f1   = 2*prec*rec/(prec+rec+1e-8)

        if f1 > best_f1:
            best_f1    = f1
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}

    clf.load_state_dict(best_state)
    clf.eval()
    probs_list, true_list = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            probs_list.extend(torch.sigmoid(clf(xb).squeeze(-1)).cpu().tolist())
            true_list.extend(yb.cpu().tolist())
    probs = np.array(probs_list)
    truth = np.array(true_list, dtype=int)
    preds = (probs > THRESHOLD).astype(int)
    return calc_metrics(preds, probs, truth)



class StandaloneLSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = 1,
            hidden_size   = 64,
            num_layers    = 2,
            batch_first   = True,
            dropout       = LSTM_DROPOUT,
            bidirectional = True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(LSTM_DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)         # (B, F, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Running {len(SEEDS)} seeds × 4 models = {len(SEEDS)*4} experiments\n")

    data = load_data()
    print(f"[Data] X_feat={data['X_feat'].shape}, y: {dict(Counter(data['y']))}\n")

    results = {
        'MLP':      {'sensitivity': [], 'specificity': [], 'mcc': [], 'auc': []},
        'LSTM':     {'sensitivity': [], 'specificity': [], 'mcc': [], 'auc': []},
        'MLP-VAE':  {'sensitivity': [], 'specificity': [], 'mcc': [], 'auc': []},
        'LSTM-VAE': {'sensitivity': [], 'specificity': [], 'mcc': [], 'auc': []},
    }

    for i, seed in enumerate(SEEDS):
        print(f"{'='*60}")
        print(f"[Seed {seed}]  ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        split    = prepare_split(data, seed)
        X_tr     = split['X_feat_tr']
        y_tr     = split['y_tr']
        X_te     = split['X_feat_te']
        y_te     = split['y_te']
        feat_dim = X_tr.shape[-1]

        print(f"  [MLP] training...", end=' ', flush=True)
        set_seed(seed)
        mlp   = StandaloneMLP(feat_dim, dropout=MLP_DROPOUT).to(DEVICE)
        sens, spec, mcc, auc_s = run_standalone(mlp, X_tr, y_tr, X_te, y_te,
                                                   MLP_EPOCHS, MLP_BATCH, MLP_LR, MLP_WD, MLP_SMOOTH)
        results['MLP']['sensitivity'].append(sens)
        results['MLP']['specificity'].append(spec)
        results['MLP']['mcc'].append(mcc)
        results['MLP']['auc'].append(auc_s)
        print(f"Sens={sens:.3f} Spec={spec:.3f} MCC={mcc:.3f} AUC={auc_s:.3f}")

        print(f"  [LSTM] training...", end=' ', flush=True)
        set_seed(seed)
        lstm_m = StandaloneLSTM(feat_dim).to(DEVICE)
        sens, spec, mcc, auc_s = run_standalone(lstm_m, X_tr, y_tr, X_te, y_te,
                                                   LSTM_EPOCHS, LSTM_BATCH, LSTM_LR, LSTM_WD, LSTM_SMOOTH)
        results['LSTM']['sensitivity'].append(sens)
        results['LSTM']['specificity'].append(spec)
        results['LSTM']['mcc'].append(mcc)
        results['LSTM']['auc'].append(auc_s)
        print(f"Sens={sens:.3f} Spec={spec:.3f} MCC={mcc:.3f} AUC={auc_s:.3f}")

        print(f"  [MLP-VAE] training...", end=' ', flush=True)
        set_seed(seed)
        sens, spec, mcc, auc_s = run_vae_model(
            mlp_create_vae, mlp_create_clf, mlp_vae_loss,
            X_tr, y_tr, X_te, y_te)
        results['MLP-VAE']['sensitivity'].append(sens)
        results['MLP-VAE']['specificity'].append(spec)
        results['MLP-VAE']['mcc'].append(mcc)
        results['MLP-VAE']['auc'].append(auc_s)
        print(f"Sens={sens:.3f} Spec={spec:.3f} MCC={mcc:.3f} AUC={auc_s:.3f}")

        print(f"  [LSTM-VAE] training...", end=' ', flush=True)
        set_seed(seed)
        sens, spec, mcc, auc_s = run_vae_model(
            lstm_create_vae, lstm_create_clf, lstm_vae_loss,
            X_tr, y_tr, X_te, y_te)
        results['LSTM-VAE']['sensitivity'].append(sens)
        results['LSTM-VAE']['specificity'].append(spec)
        results['LSTM-VAE']['mcc'].append(mcc)
        results['LSTM-VAE']['auc'].append(auc_s)
        print(f"Sens={sens:.3f} Spec={spec:.3f} MCC={mcc:.3f} AUC={auc_s:.3f}")


    with open(os.path.join(RESULTS_DIR, 'robustness_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print("Raw results saved to results/robustness_results.pkl")

    print(f"\n{'='*70}")
    print(f"{'Model':<12} {'Sensitivity':>13} {'Specificity':>13} "
          f"{'MCC':>10} {'AUC':>10}")
    print(f"{'='*70}")
    for model_name, metrics in results.items():
        for metric_name, values in metrics.items():
            arr = np.array(values)
            metrics[metric_name + '_mean'] = arr.mean()
            metrics[metric_name + '_std']  = arr.std()

        print(f"{model_name:<12}  "
              f"Sens {metrics['sensitivity_mean']:.3f}±{metrics['sensitivity_std']:.3f}  "
              f"Spec {metrics['specificity_mean']:.3f}±{metrics['specificity_std']:.3f}  "
              f"MCC {metrics['mcc_mean']:.3f}±{metrics['mcc_std']:.3f}  "
              f"AUC {metrics['auc_mean']:.3f}±{metrics['auc_std']:.3f}")
    print(f"{'='*70}")

    model_names  = ['MLP', 'LSTM', 'MLP-VAE', 'LSTM-VAE']
    metric_keys  = ['sensitivity', 'specificity', 'mcc', 'auc']
    metric_labels = ['Sensitivity', 'Specificity', 'MCC', 'AUC']

    colors = ['#5B8DB8', '#7BAFD4', '#D4813A', '#E8A96A']

    fig, axes = plt.subplots(1, 4, figsize=(18, 7))
    fig.suptitle(f'Model Robustness Comparison ({len(SEEDS)} runs)',
                 fontsize=15, fontweight='bold', y=1.01)

    for ax, metric, label in zip(axes, metric_keys, metric_labels):
        data_to_plot = [results[m][metric] for m in model_names]

        parts = ax.violinplot(data_to_plot,
                              positions=range(len(model_names)),
                              showmeans=True, showmedians=True,
                              showextrema=True)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        for component in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
            if component in parts:
                parts[component].set_color('black')
                parts[component].set_linewidth(1.2)

        for j, (vals, color) in enumerate(zip(data_to_plot, colors)):
            jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(np.full(len(vals), j) + jitter, vals,
                       color=color, edgecolors='black',
                       linewidths=0.5, s=30, zorder=3, alpha=0.9)

        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, fontsize=10, rotation=15)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

        for j, vals in enumerate(data_to_plot):
            m, s = np.mean(vals), np.std(vals)
            ax.text(j, ax.get_ylim()[0] - 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f'{m:.3f}\n±{s:.3f}',
                    ha='center', va='top', fontsize=7.5, color='#333333')


        ymin = min(min(v) for v in data_to_plot)
        ymax = max(max(v) for v in data_to_plot)
        margin = (ymax - ymin) * 0.15 if ymax > ymin else 0.05
        ax.set_ylim(ymin - margin * 2.5, ymax + margin)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, alpha=0.75, label=n)
                      for c, n in zip(colors, model_names)]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor='gray')

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'robustness_violin.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nViolin plot saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()