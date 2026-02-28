import torch
import torch.nn as nn


# ========================== LSTM FEATURE VAE ================================
# Encoder: BiLSTM + Attention Pooling
# Decoder: 单向 LSTM，hidden 由 latent 初始化

class ConditionalLSTMFeatureVAE(nn.Module):
    """
    Conditional LSTM-VAE，直接在归一化临床特征空间操作。
    把 (B, feat_dim) 视为 feat_dim 个时间步 x 1 维的序列。
    Encoder: BiLSTM + Attention -> mu / logvar
    Decoder: LSTM（hidden 由 latent 初始化）-> (B, feat_dim)
    """
    def __init__(self, feat_dim=15, hidden_dim=64, latent_dim=16,
                 num_layers=2, n_classes=2, embed_dim=8, dropout=0.2):
        super().__init__()
        self.feat_dim   = feat_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.class_embed = nn.Embedding(n_classes, embed_dim)

        # Encoder: BiLSTM
        self.encoder = nn.LSTM(
            input_size    = 1 + embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )
        enc_out_dim = hidden_dim * 2
        self.attn_w    = nn.Linear(enc_out_dim, 1)
        self.fc_mu     = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # Decoder: 单向 LSTM
        self.latent_to_h = nn.Linear(latent_dim + embed_dim, hidden_dim * num_layers)
        self.latent_to_c = nn.Linear(latent_dim + embed_dim, hidden_dim * num_layers)
        self.decoder = nn.LSTM(
            input_size  = latent_dim + embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, 1)

    def _attn_pool(self, enc_out):
        scores  = self.attn_w(enc_out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)

    def encode(self, x, labels):
        B, F    = x.shape
        emb     = self.class_embed(labels)
        emb_rep = emb.unsqueeze(1).expand(-1, F, -1)
        inp     = torch.cat([x.unsqueeze(-1), emb_rep], dim=-1)  # (B, F, 1+embed)
        enc_out, _ = self.encoder(inp)
        ctx    = self._attn_pool(enc_out)
        mu     = self.fc_mu(ctx)
        logvar = self.fc_logvar(ctx)
        std    = torch.exp(0.5 * logvar)
        z      = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z, labels):
        B      = z.size(0)
        emb    = self.class_embed(labels)
        z_cond = torch.cat([z, emb], dim=-1)
        h0 = self.latent_to_h(z_cond).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        c0 = self.latent_to_c(z_cond).view(B, self.num_layers, self.hidden_dim).permute(1,0,2).contiguous()
        z_rep      = z_cond.unsqueeze(1).expand(-1, self.feat_dim, -1)
        dec_out, _ = self.decoder(z_rep, (h0, c0))
        return self.fc_out(dec_out).squeeze(-1)  # (B, feat_dim)

    def forward(self, x, labels):
        z, mu, logvar = self.encode(x, labels)
        return self.decode(z, labels), mu, logvar

    def generate(self, class_label, num_samples=1, device='cpu'):
        self.eval()
        with torch.no_grad():
            z      = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
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
    """LSTM Feature-VAE，seq_len 保留兼容旧签名，不使用。"""
    return ConditionalLSTMFeatureVAE(
        feat_dim   = feat_dim,
        hidden_dim = 64,
        latent_dim = 16,
        num_layers = 2,
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