# Iteration 1: Answer

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
frames = np.transpose(frames, (0, 3, 1, 2))  # (N, 3, 64, 64)
frames = torch.from_numpy(frames)

print(f"Loaded {frames.shape[0]} frames, shape: {frames.shape}")

# ============================================================
# 2. Dataset — single frames, target = same frame
# ============================================================
class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]  # (3, 64, 64)
        return frame, frame       # input = target (autoencoder)

dataset = FrameDataset(frames)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ============================================================
# 3. Model
# ============================================================
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()
        # Encoder: (3,64,64) → (32,32,32) → (64,16,16) → (128,8,8) → (16,8,8)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_channels, 3, stride=1, padding=1),
        )
        # Decoder: (16,8,8) → (128,8,8) → (64,16,16) → (32,32,32) → (3,64,64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ConvAutoencoder().to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
# → ~200K parameters (vs ~6.3M for the MLP — 30x fewer!)

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

        reconstruction = model(input_batch)
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

indices = np.random.choice(len(dataset), 5, replace=False)
with torch.no_grad():
    for row, idx in enumerate(indices):
        frame = frames[idx].unsqueeze(0).to(device)  # (1, 3, 64, 64)
        recon = model(frame).cpu().squeeze(0)         # (3, 64, 64)

        orig_img = frames[idx].permute(1, 2, 0).numpy()       # (64, 64, 3)
        recon_img = recon.permute(1, 2, 0).clamp(0, 1).numpy()

        axes[row, 0].imshow(orig_img)
        axes[row, 1].imshow(recon_img)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

plt.tight_layout()
plt.savefig("reconstructions.png", dpi=150)
plt.show()

# ============================================================
# 6. Inspect the Latent Space
# ============================================================
with torch.no_grad():
    sample = frames[0].unsqueeze(0).to(device)
    latent = model.encode(sample)
    print(f"\nLatent shape: {latent.shape}")   # (1, 16, 8, 8)
    print(f"Latent min:   {latent.min().item():.3f}")
    print(f"Latent max:   {latent.max().item():.3f}")
    print(f"Latent mean:  {latent.mean().item():.3f}")
    print(f"Latent std:   {latent.std().item():.3f}")
    # → Values are arbitrary floats with no clear structure

# ============================================================
# 7. Try Decoding Random Noise
# ============================================================
with torch.no_grad():
    random_latent = torch.randn(1, 16, 8, 8).to(device)
    generated = model.decode(random_latent).cpu().squeeze(0)
    plt.figure(figsize=(3, 3))
    plt.imshow(generated.permute(1, 2, 0).clamp(0, 1).numpy())
    plt.title("Decoded from random noise")
    plt.axis("off")
    plt.savefig("random_decode.png", dpi=150)
    plt.show()
    # → Expect garbage — the latent space has no meaningful structure
```

## What You Should Observe

1. **Reconstructions are much better than Iteration 0** — the model can accurately reconstruct most frames. Convolutions exploit spatial structure effectively.

2. **~200K parameters vs ~6.3M** — 30x fewer parameters, yet much better results. That's the power of inductive biases (locality + weight sharing).

3. **Latent space is unstructured** — the values are arbitrary floats (e.g., range -5 to +8). There's no reason neighboring values should mean similar things.

4. **Random decode produces garbage** — since the latent space has no imposed structure, random points don't correspond to valid frames. You can't "sample" from this model to generate new frames.

## Why This Matters

You now have a good **compressor** but not a **generator**. For the world model, you need to:
- Predict future frame tokens (like predicting next words)
- That requires a **discrete vocabulary** of visual tokens
- Continuous floats can't be predicted with cross-entropy like language models

**Next iteration**: force the latent space to be **discrete** using Finite Scalar Quantization (FSQ), creating a visual vocabulary analogous to words.
