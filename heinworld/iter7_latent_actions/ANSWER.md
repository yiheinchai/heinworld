# Iteration 7: Answer

```python
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# 1. Load Data
# ============================================================
with h5py.File("../data/pong_frames.h5", "r") as f:
    raw_frames = f["frames"][:]

frames = raw_frames.astype(np.float32) / 255.0
frames = np.transpose(frames, (0, 3, 1, 2))
frames = torch.from_numpy(frames)

FRAME_SKIP = 2

class FramePairDataset(Dataset):
    def __init__(self, frames, frame_skip=2):
        self.frames = frames
        self.frame_skip = frame_skip

    def __len__(self):
        return len(self.frames) - self.frame_skip

    def __getitem__(self, idx):
        return self.frames[idx], self.frames[idx + self.frame_skip]

dataset = FramePairDataset(frames, FRAME_SKIP)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ============================================================
# 2. Binary FSQ
# ============================================================
class BinaryFSQ(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("basis", 2 ** torch.arange(action_dim))

    def quantize(self, z):
        z = torch.sigmoid(z)                         # [0, 1]
        z_q = z + (z.round() - z).detach()           # straight-through to {0, 1}
        return z_q

    def codes_to_indices(self, z_q):
        return (z_q * self.basis).sum(dim=-1).long()

    def indices_to_codes(self, indices):
        codes = []
        remaining = indices.clone()
        for i in range(self.action_dim - 1, -1, -1):
            codes.append(torch.div(remaining, self.basis[i], rounding_mode='floor'))
            remaining = remaining % self.basis[i]
        codes.reverse()
        return torch.stack(codes, dim=-1).float()

    def forward(self, z):
        z_q = self.quantize(z)
        indices = self.codes_to_indices(z_q)
        return z_q, indices

# ============================================================
# 3. Transformer Block (reused)
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
# 4. Latent Action Model
# ============================================================
class LatentActionModel(nn.Module):
    def __init__(self, patch_size=8, d_model=128, n_heads=4,
                 enc_layers=2, dec_layers=2, action_dim=3, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2

        # --- Action Encoder ---
        # Takes concatenated [frame_t, frame_t+1] along channel dim → 6 channels
        self.enc_patch_embed = nn.Conv2d(6, d_model, kernel_size=patch_size, stride=patch_size)
        self.enc_pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.enc_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(enc_layers)])
        self.action_head = nn.Linear(d_model, action_dim)

        # --- Binary FSQ ---
        self.fsq = BinaryFSQ(action_dim=action_dim)

        # --- Action Decoder ---
        # Takes frame_t patches + action conditioning → predicts frame_t+1
        self.dec_patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.dec_pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.dec_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(dec_layers)])
        self.pixel_head = nn.Linear(d_model, patch_size * patch_size * 3)

    def encode_action(self, frame_t, frame_t1):
        """(B, 3, H, W), (B, 3, H, W) → action_code (B, action_dim), indices (B,)"""
        x = torch.cat([frame_t, frame_t1], dim=1)          # (B, 6, H, W)
        tokens = self.enc_patch_embed(x)                     # (B, d_model, 8, 8)
        tokens = tokens.flatten(2).transpose(1, 2) + self.enc_pos  # (B, P, d_model)

        for block in self.enc_blocks:
            tokens = block(tokens)

        pooled = tokens.mean(dim=1)                          # (B, d_model)
        action_latent = self.action_head(pooled)             # (B, action_dim)
        action_code, action_idx = self.fsq(action_latent)    # (B, action_dim), (B,)
        return action_code, action_idx, action_latent

    def decode_frame(self, frame_t, action_code):
        """(B, 3, H, W), (B, action_dim) → predicted frame_t+1 (B, 3, H, W)"""
        tokens = self.dec_patch_embed(frame_t)               # (B, d_model, 8, 8)
        tokens = tokens.flatten(2).transpose(1, 2) + self.dec_pos  # (B, P, d_model)

        # Add action to all patch embeddings
        action_emb = self.action_proj(action_code).unsqueeze(1)  # (B, 1, d_model)
        tokens = tokens + action_emb  # broadcast

        for block in self.dec_blocks:
            tokens = block(tokens)

        pixels = self.pixel_head(tokens)                     # (B, P, ps*ps*3)
        pixels = torch.sigmoid(pixels)

        B = frame_t.shape[0]
        ps = self.patch_size
        gs = self.grid_size
        pixels = pixels.reshape(B, gs, gs, ps, ps, 3)
        pixels = pixels.permute(0, 5, 1, 3, 2, 4).reshape(B, 3, gs * ps, gs * ps)
        return pixels

    def forward(self, frame_t, frame_t1):
        action_code, action_idx, action_latent = self.encode_action(frame_t, frame_t1)
        predicted = self.decode_frame(frame_t, action_code)
        return predicted, action_code, action_idx, action_latent

model = LatentActionModel().to(device)
print(f"Latent Action Model params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 5. Train
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 30
for epoch in range(num_epochs):
    total_loss = 0
    total_recon = 0
    total_var = 0
    count = 0

    for frame_t, frame_t1 in loader:
        frame_t = frame_t.to(device)
        frame_t1 = frame_t1.to(device)

        predicted, action_code, action_idx, action_latent = model(frame_t, frame_t1)

        # Reconstruction loss
        recon_loss = F.l1_loss(predicted, frame_t1)

        # Variance penalty — encourage diverse actions
        var = action_latent.var(dim=0).mean()
        var_penalty = 1.0 / (var + 1e-6)

        loss = recon_loss + 0.1 * var_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_var += var.item()
        count += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/count:.4f}, "
              f"Recon: {total_recon/count:.4f}, Var: {total_var/count:.4f}")

# ============================================================
# 6. Visualize Reconstructions
# ============================================================
model.eval()
fig, axes = plt.subplots(5, 3, figsize=(9, 15))
axes[0, 0].set_title("Frame t")
axes[0, 1].set_title("Predicted t+1")
axes[0, 2].set_title("Actual t+1")

sample_indices = np.random.choice(len(dataset), 5, replace=False)
with torch.no_grad():
    for row, idx in enumerate(sample_indices):
        ft, ft1 = dataset[idx]
        ft_dev = ft.unsqueeze(0).to(device)
        ft1_dev = ft1.unsqueeze(0).to(device)
        pred, _, action_idx, _ = model(ft_dev, ft1_dev)

        axes[row, 0].imshow(ft.permute(1, 2, 0).numpy())
        axes[row, 1].imshow(pred[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[row, 2].imshow(ft1.permute(1, 2, 0).numpy())
        for c in range(3):
            axes[row, c].axis("off")
        axes[row, 1].set_ylabel(f"act={action_idx.item()}", fontsize=10)

plt.tight_layout()
plt.savefig("latent_action_recon.png", dpi=150)
plt.show()

# ============================================================
# 7. Action Distribution
# ============================================================
all_actions = []
with torch.no_grad():
    for i in range(0, min(len(dataset), 2000)):
        ft, ft1 = dataset[i]
        _, action_idx, _, _ = model(
            ft.unsqueeze(0).to(device),
            ft1.unsqueeze(0).to(device)
        )
        all_actions.append(action_idx.item())

all_actions = np.array(all_actions)
plt.figure(figsize=(8, 4))
plt.hist(all_actions, bins=np.arange(9) - 0.5, rwidth=0.8, edgecolor='black')
plt.xticks(range(8))
plt.xlabel("Action Code")
plt.ylabel("Count")
plt.title("Distribution of Discovered Action Codes")
plt.savefig("action_distribution.png", dpi=150)
plt.show()

unique_actions = len(np.unique(all_actions))
print(f"Unique actions used: {unique_actions}/8")

# ============================================================
# 8. Action Semantics — Group by action, show examples
# ============================================================
fig, axes = plt.subplots(8, 6, figsize=(18, 24))
with torch.no_grad():
    for action_id in range(8):
        matching = [i for i, a in enumerate(all_actions) if a == action_id]
        if len(matching) == 0:
            for c in range(6):
                axes[action_id, c].axis("off")
            axes[action_id, 0].set_ylabel(f"Act {action_id}\n(unused)", fontsize=10)
            continue

        examples = matching[:3]  # show 3 pairs
        axes[action_id, 0].set_ylabel(f"Act {action_id}\n(n={len(matching)})", fontsize=10)

        for j, idx in enumerate(examples):
            ft, ft1 = dataset[idx]
            axes[action_id, j * 2].imshow(ft.permute(1, 2, 0).numpy())
            axes[action_id, j * 2 + 1].imshow(ft1.permute(1, 2, 0).numpy())
            axes[action_id, j * 2].set_title("t", fontsize=8)
            axes[action_id, j * 2 + 1].set_title("t+1", fontsize=8)

        for c in range(6):
            axes[action_id, c].axis("off")

plt.suptitle("Frame pairs grouped by discovered action code", fontsize=14)
plt.tight_layout()
plt.savefig("action_semantics.png", dpi=150)
plt.show()
```

## What You Should Observe

1. **Meaningful actions emerge** — different codes capture different frame transitions. You might see code 0 = ball moving right, code 3 = paddle moving up, etc. The exact mapping varies by run, but clusters are semantically meaningful.

2. **Good codebook utilization** — with the variance penalty, most of the 8 codes are used. Without it, the model often collapses to 1-2 codes.

3. **Decent reconstruction** — given just frame_t and a 3-bit action code, the model predicts frame_t+1 reasonably well. Not perfect, but enough to capture the key changes.

4. **No labels were used** — the actions were discovered purely through the information bottleneck.

## Why This Matters

You've solved the scalability problem. Instead of needing labeled actions:
- Train the latent action model on any video → get inferred actions
- Use these inferred actions to train the dynamics model (replacing hardcoded labels from Iteration 6)
- At inference time, the user picks action codes (0-7) to control the world

**Next iteration**: replace the slow autoregressive decoding (one token at a time) with MaskGIT's parallel decoding (all tokens at once, iteratively refined).
