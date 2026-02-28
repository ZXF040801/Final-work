import torch
import torch.nn as nn


# ========================== MLP FEATURE VAE ================================
# VAE 部分与 model_MLP.py 完全相同：纯 MLP，直接在特征空间操作

class ConditionalMLPFeatureVAE(nn.Module):
    """
    Conditional VAE，Encoder / Decoder 均为纯 MLP，
    直接在归一化临床特征空间操作。
    （与 model_MLP.py 完全相同）
    """
    def __init__(self, feat_dim=15, hidden_dim=128, latent_dim=16,
                 n_classes=2, embed_dim=8, dropout=0.2):
        super().__init__()
        self.feat_dim   = feat_dim
        self.latent_dim = latent_dim

        self.class_embed = nn.Embedding(n_classes, embed_dim)

        # ── Encoder ──────────────────────────────────────────────────────
        enc_in = feat_dim + embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # ── Decoder ──────────────────────────────────────────────────────
        dec_in = latent_dim + embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def encode(self, x, labels):
        """x: (B, feat_dim), labels: (B,) → z, mu, logvar"""
        emb    = self.class_embed(labels)
        h      = self.encoder(torch.cat([x, emb], dim=-1))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std    = torch.exp(0.5 * logvar)
        z      = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z, labels):
        """z: (B, latent_dim), labels: (B,) → recon: (B, feat_dim)"""
        emb = self.class_embed(labels)
        return self.decoder(torch.cat([z, emb], dim=-1))

    def forward(self, x, labels):
        z, mu, logvar = self.encode(x, labels)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    def generate(self, class_label, num_samples=1, device='cpu'):
        """采样 z ~ N(0,I)，直接生成归一化特征向量 (num_samples, feat_dim)"""
        self.eval()
        with torch.no_grad():
            z      = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), class_label,
                                dtype=torch.long).to(device)
            return self.decode(z, labels)


def vae_loss_fn(recon, target, mu, logvar, kl_weight=0.001, free_bits=0.1):
    """MSE 重建损失 + KL（带 free bits）"""
    recon_loss = nn.functional.mse_loss(recon, target, reduction='mean')
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss    = kl_per_dim.sum(dim=-1).mean()
    total      = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


# ========================== LSTM CLASSIFIER ================================
# 分类器部分与原 model_LSTM.py 完全相同：BiLSTM + Attention

class ClinicalFeatureClassifier(nn.Module):
    """
    BiLSTM + Attention classifier on clinical features.

    Pipeline:
      (B, feat_dim) → unsqueeze → (B, feat_dim, 1)
                    → feat_proj(16) → (B, feat_dim, 16)
                    → BiLSTM(128)   → (B, feat_dim, 256)
                    → attention + last_hidden → concat(512)
                    → BN → FC(256) → FC(64) → FC(1)
    """
    def __init__(self, input_dim=15, hidden_dim=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size    = 16,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        enc_dim  = hidden_dim * 2   # 256
        fuse_dim = enc_dim * 2      # 512（attention_ctx + last_hidden）

        self.attn = nn.Linear(enc_dim, 1)

        self.head = nn.Sequential(
            nn.BatchNorm1d(fuse_dim),
            nn.Linear(fuse_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (B, feat_dim)
        x = x.unsqueeze(-1)                                           # (B, F, 1)
        x = self.feat_proj(x)                                         # (B, F, 16)
        out, _ = self.lstm(x)                                         # (B, F, 256)

        # Attention pooling
        weights  = torch.softmax(self.attn(out).squeeze(-1), dim=-1) # (B, F)
        attn_ctx = torch.bmm(weights.unsqueeze(1), out).squeeze(1)   # (B, 256)

        # Last hidden state
        last_h = out[:, -1, :]                                        # (B, 256)

        fused = torch.cat([attn_ctx, last_h], dim=-1)                 # (B, 512)
        return self.head(fused).squeeze(-1)                           # (B,)


# ========================== FACTORY ========================================

def create_vae(feat_dim=15, seq_len=None, cfg=None):
    """MLP Feature-VAE，seq_len 保留兼容旧签名，不使用。"""
    return ConditionalMLPFeatureVAE(
        feat_dim   = feat_dim,
        hidden_dim = 128,
        latent_dim = 16,
        n_classes  = 2,
        embed_dim  = 8,
        dropout    = 0.2,
    )


def create_classifier(input_dim=15, cfg=None):
    """BiLSTM + Attention 分类器。"""
    return ClinicalFeatureClassifier(
        input_dim  = input_dim,
        hidden_dim = 128,
        num_layers = 2,
        dropout    = 0.4,
    )


# ========================== TEST ===========================================

if __name__ == "__main__":
    B, F = 8, 10

    # VAE test
    vae    = create_vae(F)
    x      = torch.randn(B, F)
    labels = torch.randint(0, 2, (B,))

    recon, mu, lv = vae(x, labels)
    print(f"[VAE]  input={x.shape} → recon={recon.shape}")
    print(f"       mu={mu.shape}, logvar={lv.shape}")
    print(f"  Params: {sum(p.numel() for p in vae.parameters()):,}")

    loss, rl, kl = vae_loss_fn(recon, x, mu, lv)
    print(f"  Loss={loss.item():.4f}  Recon={rl.item():.4f}  KL={kl.item():.4f}")

    syn = vae.generate(1, num_samples=4)
    print(f"  Generate class-1: {syn.shape}")

    # Classifier test
    clf = create_classifier(F)
    out = clf(torch.randn(B, F))
    print(f"\n[CLF]  BiLSTM input={x.shape} → output={out.shape}")
    print(f"  Params: {sum(p.numel() for p in clf.parameters()):,}")