# Iteration 4: Answer

```python
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 1. Load Data
# ============================================================
with h5py.File("../data/pong_frames.h5", "r") as f:
    raw_frames = f["frames"][:]

frames = raw_frames.astype(np.float32) / 255.0
frames = np.transpose(frames, (0, 3, 1, 2))
frames = torch.from_numpy(frames)

class VideoClipDataset(Dataset):
    def __init__(self, frames, num_frames=4, frame_skip=2):
        self.frames = frames
        self.num_frames = num_frames
        self.frame_skip = frame_skip

    def __len__(self):
        return len(self.frames) - (self.num_frames * self.frame_skip)

    def __getitem__(self, idx):
        indices = range(idx, idx + self.num_frames * self.frame_skip, self.frame_skip)
        clip = self.frames[list(indices)]  # (T, 3, 64, 64)
        return clip, clip

dataset = VideoClipDataset(frames, num_frames=4, frame_skip=2)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ============================================================
# 2. FSQ (reused)
# ============================================================
class FSQ(nn.Module):
    def __init__(self, num_bins=4, latent_dim=5):
        super().__init__()
        self.num_bins = num_bins
        self.latent_dim = latent_dim
        self.register_buffer("basis", num_bins ** torch.arange(latent_dim))

    def quantize(self, z):
        z = torch.tanh(z)
        z = (z + 1) / 2 * (self.num_bins - 1)
        z_q = z + (z.round() - z).detach()
        return z_q

    def codes_to_indices(self, z_q):
        return (z_q * self.basis).sum(dim=-1).long()

    def forward(self, z):
        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)
        return z_q, indices

# ============================================================
# 3. Space-Time Transformer Block
# ============================================================
class SpaceTimeBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x, T, P, causal_mask=None):
        B_orig = x.shape[0]
        E = x.shape[-1]

        # --- Spatial attention: within each frame ---
        # (B, T*P, E) → (B*T, P, E)
        x_spatial = x.reshape(B_orig * T, P, E)
        ln_x = self.ln1(x_spatial)
        x_spatial = x_spatial + self.spatial_attn(ln_x, ln_x, ln_x)[0]
        x = x_spatial.reshape(B_orig, T * P, E)

        # --- Temporal attention: across frames per patch ---
        # (B, T*P, E) → (B, T, P, E) → (B, P, T, E) → (B*P, T, E)
        x_temp = x.reshape(B_orig, T, P, E).permute(0, 2, 1, 3).reshape(B_orig * P, T, E)
        ln_x = self.ln2(x_temp)
        x_temp = x_temp + self.temporal_attn(ln_x, ln_x, ln_x, attn_mask=causal_mask)[0]
        x = x_temp.reshape(B_orig, P, T, E).permute(0, 2, 1, 3).reshape(B_orig, T * P, E)

        # --- FFN ---
        x = x + self.ffn(self.ln3(x))
        return x

# ============================================================
# 4. Space-Time Video Tokenizer
# ============================================================
class SpaceTimeTokenizer(nn.Module):
    def __init__(self, patch_size=8, d_model=128, n_heads=4, n_layers=2,
                 latent_dim=5, num_bins=4, img_size=64, num_frames=4):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid_size = img_size // patch_size  # 8
        self.n_patches = self.grid_size ** 2      # 64
        self.num_frames = num_frames

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings (spatial + temporal, separate for encoder/decoder)
        self.enc_spatial_pos = nn.Parameter(torch.randn(1, 1, self.n_patches, d_model) * 0.02)
        self.enc_temporal_pos = nn.Parameter(torch.randn(1, num_frames, 1, d_model) * 0.02)
        self.dec_spatial_pos = nn.Parameter(torch.randn(1, 1, self.n_patches, d_model) * 0.02)
        self.dec_temporal_pos = nn.Parameter(torch.randn(1, num_frames, 1, d_model) * 0.02)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            SpaceTimeBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.latent_head = nn.Linear(d_model, latent_dim)

        # FSQ
        self.fsq = FSQ(num_bins=num_bins, latent_dim=latent_dim)

        # Decoder
        self.latent_embed = nn.Linear(latent_dim, d_model)
        self.decoder_blocks = nn.ModuleList([
            SpaceTimeBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.pixel_head = nn.Linear(d_model, patch_size * patch_size * 3)

        # Causal mask for temporal attention
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(num_frames, num_frames), diagonal=1).bool()
        )

    def _patch_embed_clip(self, clip):
        """clip: (B, T, 3, H, W) → (B, T, P, d_model)"""
        B, T = clip.shape[:2]
        # Process all frames at once
        flat = clip.reshape(B * T, 3, clip.shape[3], clip.shape[4])  # (B*T, 3, H, W)
        tokens = self.patch_embed(flat)                               # (B*T, d_model, 8, 8)
        tokens = tokens.flatten(2).transpose(1, 2)                    # (B*T, P, d_model)
        tokens = tokens.reshape(B, T, self.n_patches, self.d_model)   # (B, T, P, d_model)
        return tokens

    def _pixels_from_tokens(self, tokens):
        """tokens: (B, T, P, d_model) → (B, T, 3, H, W)"""
        B, T = tokens.shape[:2]
        pixels = self.pixel_head(tokens)              # (B, T, P, ps*ps*3)
        pixels = torch.sigmoid(pixels)

        ps = self.patch_size
        gs = self.grid_size
        pixels = pixels.reshape(B * T, gs, gs, ps, ps, 3)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4)
        pixels = pixels.reshape(B * T, 3, gs * ps, gs * ps)
        pixels = pixels.reshape(B, T, 3, gs * ps, gs * ps)
        return pixels

    def encode(self, clip):
        B, T = clip.shape[:2]
        P = self.n_patches

        tokens = self._patch_embed_clip(clip)                     # (B, T, P, E)
        tokens = tokens + self.enc_spatial_pos + self.enc_temporal_pos  # broadcast
        tokens = tokens.reshape(B, T * P, self.d_model)           # (B, T*P, E)

        for block in self.encoder_blocks:
            tokens = block(tokens, T, P, causal_mask=self.causal_mask)

        tokens = tokens.reshape(B, T, P, self.d_model)
        latents = self.latent_head(tokens)                        # (B, T, P, L)
        z_q, indices = self.fsq(latents)                          # (B, T, P, L), (B, T, P)
        return z_q, indices

    def decode(self, z_q):
        B, T, P, L = z_q.shape

        tokens = self.latent_embed(z_q)                           # (B, T, P, E)
        tokens = tokens + self.dec_spatial_pos + self.dec_temporal_pos
        tokens = tokens.reshape(B, T * P, self.d_model)

        for block in self.decoder_blocks:
            tokens = block(tokens, T, P, causal_mask=self.causal_mask)

        tokens = tokens.reshape(B, T, P, self.d_model)
        return self._pixels_from_tokens(tokens)                   # (B, T, 3, H, W)

    def forward(self, clip):
        z_q, indices = self.encode(clip)
        reconstruction = self.decode(z_q)
        return reconstruction, z_q, indices

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SpaceTimeTokenizer().to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# ============================================================
# 5. Train
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 30
for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for clip_batch, target_batch in loader:
        clip_batch = clip_batch.to(device)
        target_batch = target_batch.to(device)

        reconstruction, z_q, indices = model(clip_batch)
        loss = F.smooth_l1_loss(reconstruction, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/count:.6f}")

# ============================================================
# 6. Visualize — 4 original frames on top, 4 reconstructions below
# ============================================================
model.eval()
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes[0, 0].set_ylabel("Original", fontsize=12)
axes[1, 0].set_ylabel("Reconstruction", fontsize=12)

with torch.no_grad():
    idx = np.random.randint(len(dataset))
    clip, _ = dataset[idx]
    clip = clip.unsqueeze(0).to(device)  # (1, 4, 3, 64, 64)
    recon, _, indices = model(clip)

    for t in range(4):
        axes[0, t].imshow(clip[0, t].cpu().permute(1, 2, 0).numpy())
        axes[0, t].set_title(f"Frame {t}")
        axes[0, t].axis("off")

        axes[1, t].imshow(recon[0, t].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, t].axis("off")

plt.suptitle("Video Clip: Original vs Reconstruction", fontsize=14)
plt.tight_layout()
plt.savefig("spacetime_reconstructions.png", dpi=150)
plt.show()

# ============================================================
# 7. Print Token Grids for Each Frame
# ============================================================
with torch.no_grad():
    print("\nToken grids per frame (8×8, vocab=1024):")
    for t in range(4):
        grid = indices[0, t].reshape(8, 8).cpu().numpy()
        print(f"\nFrame {t}:")
        print(grid)
```

## What You Should Observe

1. **All 4 frames are reconstructed** — the model handles video, not just individual frames.

2. **Token grids are temporally coherent** — consecutive frames have similar token patterns (most of the scene is static in Pong; only the ball and paddles move).

3. **You now have the complete Video Tokenizer** — this is Stage 1 of TinyWorlds. It compresses video clips into discrete tokens: each frame becomes 64 tokens from a 1024-word vocabulary.

4. **The encoding**: a 4-frame clip is now `(4, 64)` = 256 discrete tokens. Just like a 256-word paragraph.

## Why This Matters

You have a working video compressor. But compression is not generation. To build a world model, you need to **predict** what comes next.

The setup is now exactly like language modeling:
- You have a vocabulary (1024 visual tokens)
- You have sequences (frames as sequences of tokens)
- You need to predict the next tokens in the sequence

**Next iteration**: train a dynamics model that predicts future frame tokens — applying your GPT experience directly to video tokens.
