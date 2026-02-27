import math
import torch
import torch.nn as nn


# ========================== POSITIONAL ENCODING ============================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for decoder."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]


# ========================== LSTM-VAE =======================================

class ConditionalLSTMVAE(nn.Module):
    """
    Conditional LSTM-VAE for 6-channel time series.

    Fixes for reconstruction quality:
    - Encoder: bidirectional LSTM, uses all hidden states (not just last)
    - Decoder: gets positional encoding + latent at each timestep
    - Larger latent dim (64) to preserve more information
    """
    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=64,
                 num_layers=2, seq_len=300, n_classes=2, embed_dim=8,
                 dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Class embedding
        self.class_embed = nn.Embedding(n_classes, embed_dim)

        # === ENCODER ===
        self.encoder = nn.LSTM(
            input_size=input_dim + embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        enc_out_dim = hidden_dim * 2  # bidirectional

        # Attention pooling over encoder outputs
        self.attn_w = nn.Linear(enc_out_dim, 1)

        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # === DECODER ===
        # Positional encoding so decoder knows which timestep
        self.pos_enc = PositionalEncoding(d_model=16, max_len=seq_len + 10)
        dec_input_dim = latent_dim + embed_dim + 16  # z + class + position

        self.decoder = nn.LSTM(
            input_size=dec_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Initialize decoder hidden from latent
        self.latent_to_h = nn.Linear(latent_dim + embed_dim, hidden_dim * num_layers)
        self.latent_to_c = nn.Linear(latent_dim + embed_dim, hidden_dim * num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def _attention_pool(self, encoder_outputs):
        """Weighted average of encoder outputs using attention."""
        # encoder_outputs: (B, T, enc_out_dim)
        scores = self.attn_w(encoder_outputs).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1)             # (B, T)
        context = torch.bmm(weights.unsqueeze(1),
                            encoder_outputs).squeeze(1)      # (B, enc_out_dim)
        return context

    def encode(self, x, labels):
        """x: (B, T, C), labels: (B,) → z, mu, logvar"""
        B, T, _ = x.shape
        emb = self.class_embed(labels)
        emb_rep = emb.unsqueeze(1).expand(-1, T, -1)
        x_cond = torch.cat([x, emb_rep], dim=-1)

        enc_out, _ = self.encoder(x_cond)      # (B, T, hidden*2)
        context = self._attention_pool(enc_out)  # (B, hidden*2)

        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def decode(self, z, labels):
        """z: (B, latent_dim), labels: (B,) → recon: (B, T, C)"""
        B = z.size(0)
        emb = self.class_embed(labels)          # (B, embed_dim)
        z_cond = torch.cat([z, emb], dim=-1)    # (B, latent+embed)

        # Initialize hidden states from latent
        h0 = self.latent_to_h(z_cond)           # (B, hidden*num_layers)
        h0 = h0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = self.latent_to_c(z_cond)
        c0 = c0.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        # Build decoder input: z_cond + positional encoding at each step
        pos = self.pos_enc(self.seq_len).expand(B, -1, -1)  # (B, T, 16)
        z_rep = z_cond.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, T, latent+embed)
        dec_input = torch.cat([z_rep, pos], dim=-1)          # (B, T, latent+embed+16)

        dec_out, _ = self.decoder(dec_input, (h0, c0))       # (B, T, hidden)
        return self.fc_out(dec_out)                           # (B, T, input_dim)

    def forward(self, x, labels):
        z, mu, logvar = self.encode(x, labels)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    def generate(self, class_label, num_samples=1, device='cpu'):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), class_label,
                                dtype=torch.long).to(device)
            return self.decode(z, labels)


def vae_loss_fn(recon, target, mu, logvar, kl_weight=0.001, free_bits=0.1):
    """VAE loss with free bits."""
    recon_loss = nn.functional.mse_loss(recon, target, reduction='mean')

    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.sum(dim=-1).mean()

    total = recon_loss + kl_weight * kl_loss
    return total, recon_loss, kl_loss


# ========================== CLASSIFIER =====================================

class ClinicalFeatureClassifier(nn.Module):
    """
    MLP classifier. Supports both:
      - 15 clinical features  → hidden 64 → 32 → 1
      - 79 combined features  → hidden 128 → 64 → 1  (clinical + latent)
    hidden1 / hidden2 auto-scaled by create_classifier.
    """
    def __init__(self, input_dim=15, hidden1=64, hidden2=32, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ========================== FACTORY ========================================

def create_vae(input_dim=6, seq_len=300, cfg=None):
    return ConditionalLSTMVAE(
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=64,
        num_layers=2,
        seq_len=seq_len,
        n_classes=2,
        embed_dim=8,
        dropout=0.2,
    )


def create_classifier(input_dim=15, cfg=None):
    """Auto-scale hidden dims based on input size."""
    if input_dim > 20:
        # Combined clinical + latent features: wider first layer
        return ClinicalFeatureClassifier(input_dim=input_dim,
                                         hidden1=128, hidden2=64, dropout=0.4)
    return ClinicalFeatureClassifier(input_dim=input_dim,
                                     hidden1=64, hidden2=32, dropout=0.4)


# ========================== TEST ===========================================

if __name__ == "__main__":
    B, T, C = 8, 300, 6
    vae = create_vae(C, T)
    x = torch.randn(B, T, C)
    labels = torch.randint(0, 2, (B,))
    recon, mu, lv = vae(x, labels)
    print(f"[VAE]  input={x.shape} → recon={recon.shape}, "
          f"latent=({mu.shape}, {lv.shape})")
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"  Total params: {total_params:,}")

    syn = vae.generate(1, num_samples=4)
    print(f"  Generate: {syn.shape}")

    clf = create_classifier(15)
    feat = torch.randn(B, 15)
    out = clf(feat)
    print(f"\n[CLF]  input={feat.shape} → output={out.shape}")