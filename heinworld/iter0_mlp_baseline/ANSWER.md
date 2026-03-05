# Iteration 0: Answer

```python
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 1. Load Data
# ============================================================
with h5py.File("../data/pong_frames.h5", "r") as f:
    raw_frames = f["frames"][:]  # (N, 64, 64, 3), uint8

# Normalize to [0, 1] float32, rearrange to channels-first
frames = raw_frames.astype(np.float32) / 255.0              # (N, 64, 64, 3)
frames = np.transpose(frames, (0, 3, 1, 2))                 # (N, 3, 64, 64)
frames = torch.from_numpy(frames)                            # torch tensor

print(f"Loaded {frames.shape[0]} frames, shape: {frames.shape}")

# ============================================================
# 2. Dataset — returns (frame_t_flat, frame_t+1_flat) pairs
# ============================================================
class FramePairDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames  # (N, 3, 64, 64)

    def __len__(self):
        return len(self.frames) - 1  # N-1 pairs

    def __getitem__(self, idx):
        current = self.frames[idx].reshape(-1)      # (12288,)
        next_frame = self.frames[idx + 1].reshape(-1)  # (12288,)
        return current, next_frame

dataset = FramePairDataset(frames)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ============================================================
# 3. Model — simple 2-layer MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, dim=12288, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MLP().to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")
# → ~6.3 million parameters for 64x64. For 128x128, this would be ~25 million.

# ============================================================
# 4. Train
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.SmoothL1Loss()

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for input_batch, target_batch in loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        pred = model(input_batch)
        loss = loss_fn(pred, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# ============================================================
# 5. Visualize — input | predicted | actual
# ============================================================
model.eval()
fig, axes = plt.subplots(5, 3, figsize=(9, 15))
axes[0, 0].set_title("Input (frame t)")
axes[0, 1].set_title("Predicted (frame t+1)")
axes[0, 2].set_title("Actual (frame t+1)")

indices = np.random.choice(len(dataset), 5, replace=False)
with torch.no_grad():
    for row, idx in enumerate(indices):
        input_flat, target_flat = dataset[idx]

        pred_flat = model(input_flat.unsqueeze(0).to(device)).cpu().squeeze(0)

        # Reshape back to (3, 64, 64) then permute to (64, 64, 3) for display
        input_img = input_flat.reshape(3, 64, 64).permute(1, 2, 0).numpy()
        pred_img = pred_flat.reshape(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy()
        target_img = target_flat.reshape(3, 64, 64).permute(1, 2, 0).numpy()

        axes[row, 0].imshow(input_img)
        axes[row, 1].imshow(pred_img)
        axes[row, 2].imshow(target_img)

        for col in range(3):
            axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("predictions.png", dpi=150)
plt.show()
```

## What You Should Observe

1. **Predictions are blurry** — the model predicts the *average* of all possible next frames. Since the ball and paddles could be in many places, it hedges its bets and produces a smeared-out blob. This is a fundamental limitation of pixel-space regression.

2. **~6.3 million parameters** — and that's just for 64×64. The first linear layer alone is `12288 × 256 = 3.1M` parameters. For 128×128 frames, the input would be 49,152 dims → ~25M parameters. It doesn't scale.

3. **No spatial understanding** — the model treats pixel (0,0) and pixel (63,63) identically. It doesn't know that nearby pixels are related. A white pixel in the top-left and a white pixel in the bottom-right have no spatial relationship in this model.

## Why This Matters

These three problems all have the same root cause: **the model has no inductive bias for spatial structure**. Images aren't random bags of numbers — nearby pixels are correlated, objects are spatially coherent, and the same object can appear at different positions.

**Next iteration**: use convolutional layers that exploit spatial locality, and an autoencoder architecture that compresses frames into a small latent space before predicting anything.
