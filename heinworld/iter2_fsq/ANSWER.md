# Iteration 2: Answer

```python
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 1. Load Data (same as before)
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
# 2. FSQ Module
# ============================================================
class FSQ(nn.Module):
    def __init__(self, num_bins=4, latent_dim=5):
        super().__init__()
        self.num_bins = num_bins
        self.latent_dim = latent_dim
        # Basis for converting L values to a single index
        self.register_buffer(
            "basis", num_bins ** torch.arange(latent_dim)
        )  # [1, 4, 16, 64, 256] for num_bins=4, L=5

    def quantize(self, z):
        """z: (B, L, H, W) → z_q: (B, L, H, W) with values in {0, ..., num_bins-1}"""
        z = torch.tanh(z)                                # [-1, 1]
        z = (z + 1) / 2 * (self.num_bins - 1)           # [0, num_bins-1]
        z_q = z + (z.round() - z).detach()               # straight-through
        return z_q

    def codes_to_indices(self, z_q):
        """z_q: (B, L, H, W) → indices: (B, H, W) each in [0, num_bins^L - 1]"""
        # Move L dim to last: (B, H, W, L)
        z_q = z_q.permute(0, 2, 3, 1)
        indices = (z_q * self.basis).sum(dim=-1).long()
        return indices

    def forward(self, z):
        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)
        return z_q, indices

# ============================================================
# 3. Model
# ============================================================
class FSQAutoencoder(nn.Module):
    def __init__(self, latent_dim=5, num_bins=4):
        super().__init__()
        self.fsq = FSQ(num_bins=num_bins, latent_dim=latent_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, latent_dim, 3, stride=1, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        z = self.encoder(x)           # (B, L, 8, 8) continuous
        z_q, indices = self.fsq(z)    # (B, L, 8, 8) discrete, (B, 8, 8) indices
        return z_q, indices

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z_q, indices = self.encode(x)
        reconstruction = self.decode(z_q)
        return reconstruction, z_q, indices

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = FSQAutoencoder().to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# ============================================================
# 4. Train
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20
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
# 5. Visualize Reconstructions
# ============================================================
model.eval()
fig, axes = plt.subplots(5, 2, figsize=(6, 15))
axes[0, 0].set_title("Original")
axes[0, 1].set_title("Reconstruction")

indices_list = np.random.choice(len(dataset), 5, replace=False)
with torch.no_grad():
    for row, idx in enumerate(indices_list):
        frame = frames[idx].unsqueeze(0).to(device)
        recon, z_q, tok_indices = model(frame)

        orig_img = frames[idx].permute(1, 2, 0).numpy()
        recon_img = recon.cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()

        axes[row, 0].imshow(orig_img)
        axes[row, 1].imshow(recon_img)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

plt.tight_layout()
plt.savefig("fsq_reconstructions.png", dpi=150)
plt.show()

# ============================================================
# 6. Inspect Quantized Latents
# ============================================================
with torch.no_grad():
    frame = frames[0].unsqueeze(0).to(device)
    recon, z_q, tok_indices = model(frame)

    print(f"\nQuantized latent shape: {z_q.shape}")       # (1, 5, 8, 8)
    print(f"Unique values in z_q: {torch.unique(z_q).cpu().numpy()}")
    # → Should be exactly {0, 1, 2, 3}

    print(f"\nToken indices shape: {tok_indices.shape}")   # (1, 8, 8)
    print(f"Token index range: [{tok_indices.min()}, {tok_indices.max()}]")
    print(f"Vocab size: {4**5} = 1024")
    print(f"\n8x8 token grid for first frame:")
    print(tok_indices[0].cpu().numpy())

# ============================================================
# 7. Codebook Utilization
# ============================================================
all_indices = []
with torch.no_grad():
    for i in range(0, min(len(frames), 1000), 64):
        batch = frames[i:i+64].to(device)
        _, _, idx = model(batch)
        all_indices.append(idx.cpu())

all_indices = torch.cat(all_indices).flatten()
unique_codes = len(torch.unique(all_indices))
print(f"\nCodebook utilization: {unique_codes}/1024 codes used ({100*unique_codes/1024:.1f}%)")
# → FSQ should use most of the codebook (high utilization)
```

## What You Should Observe

1. **Reconstruction quality is comparable to Iteration 1** — the discrete bottleneck doesn't hurt much. You're storing just 5 integers per spatial position instead of 16 floats.

2. **All values are exactly {0, 1, 2, 3}** — the quantization is working. These are genuinely discrete.

3. **The 8×8 token grid** — each frame is now 64 discrete tokens, like a 64-word sentence. Different frames produce different token grids. Similar frames (e.g., consecutive Pong frames) produce similar grids.

4. **High codebook utilization** — FSQ typically uses 80-95% of codes. VQ-VAE often collapses to using only 10-20% of its codebook.

## Why This Matters

You now have a **visual tokenizer** — a way to convert images to discrete tokens and back. This is the foundation for treating video prediction as language modeling:

```
Text:   "The cat sat" → predict "on"     (cross-entropy over word vocabulary)
Video:  [tok_42, tok_17, tok_899, ...] → predict [tok_55, tok_23, ...]  (cross-entropy over visual vocabulary)
```

**But** you're still using CNNs, which have limited receptive fields and can't capture long-range dependencies. And you can't process multiple frames yet.

**Next iteration**: replace the CNN with a Vision Transformer (patch embeddings + transformer blocks), which will eventually let you handle video with space-time attention.
