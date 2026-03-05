# Iteration 3: Answer

```python
import h5py
import math
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

class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
    def __len__(self):
        return len(self.frames)
    def __getitem__(self, idx):
        return self.frames[idx], self.frames[idx]

dataset = FrameDataset(frames)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ============================================================
# 2. FSQ (reused from Iteration 2)
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
        indices = (z_q * self.basis).sum(dim=-1).long()
        return indices

    def forward(self, z):
        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)
        return z_q, indices

# ============================================================
# 3. Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        ln_x = self.ln1(x)
        x = x + self.attn(ln_x, ln_x, ln_x)[0]
        x = x + self.ffn(self.ln2(x))
        return x

# ============================================================
# 4. ViT Tokenizer Model
# ============================================================
class ViTTokenizer(nn.Module):
    def __init__(self, patch_size=8, d_model=128, n_heads=4, n_layers=2,
                 latent_dim=5, num_bins=4, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid_size = img_size // patch_size  # 8
        self.n_patches = self.grid_size ** 2      # 64

        # Patch embedding: image → patch tokens
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings (learned)
        self.enc_pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.dec_pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.latent_head = nn.Linear(d_model, latent_dim)

        # FSQ
        self.fsq = FSQ(num_bins=num_bins, latent_dim=latent_dim)

        # Decoder
        self.latent_embed = nn.Linear(latent_dim, d_model)
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        # Pixel head: each token → patch_size * patch_size * 3 values
        self.pixel_head = nn.Linear(d_model, patch_size * patch_size * 3)

    def encode(self, x):
        B = x.shape[0]
        # Patch embed: (B, 3, 64, 64) → (B, d_model, 8, 8) → (B, 64, d_model)
        tokens = self.patch_embed(x)                           # (B, d_model, 8, 8)
        tokens = tokens.flatten(2).transpose(1, 2)            # (B, 64, d_model)
        tokens = tokens + self.enc_pos_embed

        for block in self.encoder_blocks:
            tokens = block(tokens)

        latents = self.latent_head(tokens)                    # (B, 64, 5)
        z_q, indices = self.fsq(latents)                      # (B, 64, 5), (B, 64)
        return z_q, indices

    def decode(self, z_q):
        B = z_q.shape[0]
        tokens = self.latent_embed(z_q)                       # (B, 64, d_model)
        tokens = tokens + self.dec_pos_embed

        for block in self.decoder_blocks:
            tokens = block(tokens)

        pixels = self.pixel_head(tokens)                      # (B, 64, 192)
        pixels = torch.sigmoid(pixels)

        # Reshape to image: (B, 64, 192) → (B, 3, 64, 64)
        ps = self.patch_size
        gs = self.grid_size
        pixels = pixels.reshape(B, gs, gs, ps, ps, 3)        # (B, 8, 8, 8, 8, 3)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4)            # (B, 3, 8, 8, 8, 8)
        pixels = pixels.reshape(B, 3, gs * ps, gs * ps)      # (B, 3, 64, 64)
        return pixels

    def forward(self, x):
        z_q, indices = self.encode(x)
        reconstruction = self.decode(z_q)
        return reconstruction, z_q, indices

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ViTTokenizer().to(device)

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
    for input_batch, target_batch in loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        reconstruction, z_q, indices = model(input_batch)
        loss = F.smooth_l1_loss(reconstruction, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/count:.6f}")

# ============================================================
# 6. Visualize Reconstructions
# ============================================================
model.eval()
fig, axes = plt.subplots(5, 2, figsize=(6, 15))
axes[0, 0].set_title("Original")
axes[0, 1].set_title("Reconstruction")

sample_indices = np.random.choice(len(dataset), 5, replace=False)
with torch.no_grad():
    for row, idx in enumerate(sample_indices):
        frame = frames[idx].unsqueeze(0).to(device)
        recon, _, _ = model(frame)

        axes[row, 0].imshow(frames[idx].permute(1, 2, 0).numpy())
        axes[row, 1].imshow(recon.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy())
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

plt.tight_layout()
plt.savefig("vit_reconstructions.png", dpi=150)
plt.show()

# ============================================================
# 7. Print Token Grid
# ============================================================
with torch.no_grad():
    frame = frames[0].unsqueeze(0).to(device)
    _, _, indices = model(frame)
    grid = indices.reshape(8, 8).cpu().numpy()
    print(f"\n8×8 token grid (vocab size = 1024):")
    print(grid)

    unique = len(torch.unique(indices))
    print(f"Unique tokens in this frame: {unique}/64")
```

## What You Should Observe

1. **Reconstruction quality is comparable to the CNN version** — for single frames, transformers don't have a dramatic advantage. The benefit comes when we add temporal modeling.

2. **Token grids look structured** — nearby patches often have similar token indices because they encode similar visual content.

3. **The architecture is now transformer-based** — same as your GPT, but processing image patches instead of word tokens. The encoder is like BERT (bidirectional), the decoder too.

4. **You now have TinyWorlds' Video Tokenizer architecture** — just for a single frame. Extending to video (multiple frames) requires adding temporal attention.

## Why This Matters

You've bridged from CNNs to transformers for vision. The architecture is:
- **Patch embed**: the "tokenizer" (image → tokens)
- **Transformer encoder**: processes tokens (like GPT layers)
- **FSQ**: discretizes (like rounding to vocabulary IDs)
- **Transformer decoder**: reconstructs from discrete tokens

**The limitation**: each frame is processed independently. In a video, frames are highly correlated — frame_t and frame_t+1 share most of their content. You need **temporal attention** to capture this, but full attention over all patches across all frames is O((T×P)²).

**Next iteration**: factor attention into spatial (within frame) and temporal (across frames) components — the Space-Time Transformer.
