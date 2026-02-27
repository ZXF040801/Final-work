import os, pickle, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay,
)
from preprocessing import patient_aware_split, compute_normalization_stats, normalize

# ── 配置（与 train.py 分类器保持一致）─────────────────────────────────────
SEED        = 42
DATA_PATH   = 'preprocessed/preprocessed_data.pkl'
CKPT_DIR    = 'checkpoints'
RESULTS_DIR = 'results'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型结构
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.4    # 与 ClinicalFeatureClassifier dropout=0.4 一致

# 训练超参 —— 与 train.py CLF_* 完全对齐
EPOCHS        = 200   # CLF_EPOCHS = 200
BATCH         = 32    # CLF_BATCH  = 32
LR            = 1e-3  # CLF_LR     = 1e-3
WD            = 1e-2  # CLF_WD     = 1e-2
LABEL_SMOOTH  = 0.1   # LABEL_SMOOTH = 0.1
PATIENCE      = 40    # PATIENCE   = 40


# ── 与 train.py 相同的 LabelSmoothingBCELoss ──────────────────────────────
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, s=0.1):
        super().__init__()
        self.s = s

    def forward(self, logits, targets):
        targets = targets.float() * (1 - self.s) + 0.5 * self.s
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


# ── 模型 ──────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    """
    双向LSTM直接对6通道原始时序分类。
    分类头结构与 ClinicalFeatureClassifier 相近：
      Linear→BN→ReLU→Dropout→Linear→BN→ReLU→Dropout→Linear(1)
    """
    def __init__(self, input_dim=6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = HIDDEN_DIM,
            num_layers    = NUM_LAYERS,
            batch_first   = True,
            dropout       = DROPOUT,
            bidirectional = True,
        )
        enc_dim = HIDDEN_DIM * 2   # 双向

        # 分类头：仿照 ClinicalFeatureClassifier (64→32→1)
        self.head = nn.Sequential(
            nn.Linear(enc_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, 15) → reshape → (B, 15, 1) 每个特征作一个时间步
        if x.dim() == 2:
            x = x.unsqueeze(-1)          # (B, 15, 1)
        out, _ = self.lstm(x)           # (B, 15, hidden*2)
        last   = out[:, -1, :]          # 最后时间步 (B, hidden*2)
        return self.head(last).squeeze(-1)


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 数据加载（与 train.py 完全相同的 split）
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    X_raw  = data['X_raw']
    y      = data['y']
    pids   = data['patient_ids']
    fids   = data.get('file_ids', None)
    X_feat = data['X_feat']

    print(f"[Load] X_raw={X_raw.shape}, y: {dict(Counter(y))}, "
          f"patients={len(set(pids))}")
    split = patient_aware_split(X_raw, X_feat, y, pids, fids)
    print(f"[Split] Train: {len(split['y_tr'])} ({dict(Counter(split['y_tr']))})")
    print(f"[Split] Test:  {len(split['y_te'])} ({dict(Counter(split['y_te']))})")

    # 归一化（与 train.py 相同）
    # 使用15个临床特征（与 train.py 分类器输入相同）
    feat_mean = split['X_feat_tr'].mean(axis=0)
    feat_std  = split['X_feat_tr'].std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_tr = (split['X_feat_tr'] - feat_mean) / feat_std  # (N, 15)
    X_te = (split['X_feat_te'] - feat_mean) / feat_std
    y_tr, y_te = split['y_tr'], split['y_te']

    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)),
        batch_size=BATCH, shuffle=True)
    test_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te)),
        batch_size=BATCH)

    # 模型
    # 15个特征作为长度15、通道数1的序列送入LSTM
    model = LSTMClassifier(input_dim=1).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] BiLSTM | hidden={HIDDEN_DIM}x{NUM_LAYERS} "
          f"| params={n_params:,} | device={DEVICE}")

    # 优化器与调度（与 train.py 完全一致）
    criterion = LabelSmoothingBCELoss(LABEL_SMOOTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    print(f"\n{'='*60}")
    print(f"TRAINING LSTM Classifier")
    print(f"  epochs={EPOCHS} | batch={BATCH} | lr={LR} | wd={WD}")
    print(f"  label_smooth={LABEL_SMOOTH} | patience={PATIENCE}")
    print("  Input: 15 clinical features (same as train.py classifier)")
    print("  (params aligned with train.py for fair comparison)")
    print(f"{'='*60}")

    history  = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1  = 0.0
    patience = 0
    ckpt     = os.path.join(CKPT_DIR, 'lstm_best.pt')

    for epoch in range(EPOCHS):
        t0 = time.time()

        # 训练
        model.train()
        tr_loss, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb); n += len(yb)
        scheduler.step()

        # 验证
        model.eval()
        probs_list, true_list, vl, vn = [], [], 0.0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                vl  += criterion(logits, yb).item() * len(yb); vn += len(yb)
                probs_list.extend(torch.sigmoid(logits).cpu().tolist())
                true_list.extend(yb.cpu().tolist())

        probs = np.array(probs_list)
        truth = np.array(true_list, dtype=int)
        preds = (probs > 0.5).astype(int)
        acc   = (preds == truth).mean()
        tp = ((preds==1)&(truth==1)).sum()
        fp = ((preds==1)&(truth==0)).sum()
        fn = ((preds==0)&(truth==1)).sum()
        prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
        f1   = 2*prec*rec/(prec+rec+1e-8)

        history['train_loss'].append(tr_loss/n)
        history['val_loss'].append(vl/vn)
        history['val_acc'].append(float(acc))
        history['val_f1'].append(float(f1))

        if f1 > best_f1:
            best_f1 = f1; patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1}"); break

        if (epoch+1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {tr_loss/n:.4f} | "
                  f"Acc: {acc:.3f} | F1: {f1:.3f} | "
                  f"Rec(PD): {rec:.3f} | {time.time()-t0:.1f}s")

    # ── 最终评估 ──────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    probs_list, true_list = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            probs_list.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().tolist())
            true_list.extend(yb.tolist())

    probs = np.array(probs_list)
    truth = np.array(true_list, dtype=int)
    preds = (probs > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(truth, probs)
    auc_score   = auc(fpr, tpr)

    print(f"\n{'='*60}")
    print(f"LSTM CLASSIFIER — Test Results")
    print(f"{'='*60}")
    print(classification_report(truth, preds,
          target_names=['Non-PD (0)', 'PD (1)'], digits=3, zero_division=0))
    print(f"  AUC: {auc_score:.3f}")
    print(f"{'='*60}")

    # ── 图表 ──────────────────────────────────────────────────────────────
    cm = confusion_matrix(truth, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=['Non-PD (0)', 'PD (1)']).plot(
        ax=ax, cmap='Blues', values_format='d')
    ax.set_title('LSTM Confusion Matrix')
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, 'lstm_confusion_matrix.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {p}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'LSTM (AUC={auc_score:.3f})')
    ax.plot([0,1],[0,1],'k--',alpha=0.5)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — LSTM'); ax.legend(loc='lower right')
    p = os.path.join(RESULTS_DIR, 'lstm_roc_curve.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {p}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('LSTM Training History', fontsize=13)
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val', ls='--')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(history['val_acc']); axes[1].set_title('Val Accuracy')
    axes[2].plot(history['val_f1']);  axes[2].set_title('Val F1 (PD)')
    for ax in axes: ax.set_xlabel('Epoch')
    fig.tight_layout()
    p = os.path.join(RESULTS_DIR, 'lstm_training_curves.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {p}")

if __name__ == "__main__":
    main()