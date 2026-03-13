import torch
import torch.nn as nn



class ConditionalMLPFeatureVAE(nn.Module):
    def __init__(self, feat_dim=15, hidden_dim=128, latent_dim=16,
                 n_classes=2, embed_dim=8, dropout=0.2):
        super().__init__()
        self.feat_dim   = feat_dim
        self.latent_dim = latent_dim

        self.class_embed = nn.Embedding(n_classes, embed_dim)

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
        emb    = self.class_embed(labels)
        h      = self.encoder(torch.cat([x, emb], dim=-1))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std    = torch.exp(0.5 * logvar)
        z      = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z, labels):
        emb = self.class_embed(labels)
        return self.decoder(torch.cat([z, emb], dim=-1))

    def forward(self, x, labels):
        z, mu, logvar = self.encode(x, labels)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    def generate(self, class_label, num_samples=1, device='cpu'):
        self.eval()
        with torch.no_grad():
            z      = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), class_label,
                                dtype=torch.long).to(device)
            return self.decode(z, labels)


def vae_loss_fn(recon, target, mu, logvar, kl_weight=0.001, free_bits=0.1):
    recon_loss = nn.functional.mse_loss(recon, target, reduction='mean')
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss    = kl_per_dim.sum(dim=-1).mean()
    total      = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss



class ClinicalFeatureClassifier(nn.Module):
    def __init__(self, input_dim=15, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ========================== FACTORY ========================================

def create_vae(feat_dim=15, seq_len=None, cfg=None):
    return ConditionalMLPFeatureVAE(
        feat_dim   = feat_dim,
        hidden_dim = 128,
        latent_dim = 16,
        n_classes  = 2,
        embed_dim  = 8,
        dropout    = 0.2,
    )


def create_classifier(input_dim=15, cfg=None):
    return ClinicalFeatureClassifier(
        input_dim = input_dim,
        dropout   = 0.4,
    )


if __name__ == "__main__":
    B, F = 8, 10

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

    clf = create_classifier(F)
    out = clf(torch.randn(B, F))
    print(f"\n[CLF]  MLP input={x.shape} → output={out.shape}")
    print(f"  Params: {sum(p.numel() for p in clf.parameters()):,}")