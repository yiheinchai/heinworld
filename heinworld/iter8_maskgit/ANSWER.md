# Iteration 8: Answer

```python
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# 1. Load data, frozen tokenizer, frozen action model
# ============================================================
# (Load frames, tokenizer, action model as in previous iterations)
# tokenizer = ...  (frozen, from Iter 4)
# action_model = ... (frozen, from Iter 7)

VOCAB_SIZE = 1024
N_PATCHES = 64
MASK_ID = VOCAB_SIZE  # special mask token = 1024 (outside normal vocab)

# ============================================================
# 2. Tokenize data with inferred actions
# ============================================================
# For each clip of T frames:
#   video_tokens: (T, P) — from frozen tokenizer
#   actions: (T-1,) — from frozen action model on consecutive pairs
# Combine into dataset

# ============================================================
# 3. FiLM + Space-Time Block (from Iter 6)
# ============================================================
class FiLM(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.scale = nn.Linear(cond_dim, feature_dim)
        self.shift = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        gamma = 1 + self.scale(cond).unsqueeze(1)
        beta = self.shift(cond).unsqueeze(1)
        return gamma * x + beta

class SpaceTimeBlockFiLM(nn.Module):
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
        self.film = FiLM(d_model, d_model)

    def forward(self, x, T, P, action_embed, causal_mask=None):
        B_orig, _, E = x.shape

        # Spatial attention
        x_s = x.reshape(B_orig * T, P, E)
        ln_x = self.ln1(x_s)
        x_s = x_s + self.spatial_attn(ln_x, ln_x, ln_x)[0]
        x = x_s.reshape(B_orig, T * P, E)

        # Temporal attention (causal)
        x_t = x.reshape(B_orig, T, P, E).permute(0, 2, 1, 3).reshape(B_orig * P, T, E)
        ln_x = self.ln2(x_t)
        x_t = x_t + self.temporal_attn(ln_x, ln_x, ln_x, attn_mask=causal_mask)[0]
        x = x_t.reshape(B_orig, P, T, E).permute(0, 2, 1, 3).reshape(B_orig, T * P, E)

        # FFN + FiLM
        ffn_out = self.ffn(self.ln3(x))
        ffn_out = self.film(ffn_out, action_embed)
        x = x + ffn_out
        return x

# ============================================================
# 4. MaskGIT Dynamics Model
# ============================================================
class MaskGITDynamics(nn.Module):
    def __init__(self, vocab_size=1024, num_actions=8, d_model=128,
                 n_heads=4, n_layers=4, n_patches=64, max_frames=4):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches
        self.vocab_size = vocab_size

        # Token embeddings: vocab + 1 for [MASK]
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)
        self.action_embed = nn.Embedding(num_actions, d_model)

        # Positional embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_patches, d_model) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpaceTimeBlockFiLM(d_model, n_heads) for _ in range(n_layers)
        ])

        # Output head: predict vocab (not vocab+1, since we never predict [MASK])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_frames, max_frames), diagonal=1).bool()
        )

    def forward(self, tokens, actions):
        """
        tokens: (B, T, P) — may contain MASK_ID at some positions
        actions: (B, T)
        → logits: (B, T, P, vocab_size)
        """
        B, T, P = tokens.shape

        x = self.token_embed(tokens)  # (B, T, P, d_model)
        x = x + self.spatial_pos[:, :, :P] + self.temporal_pos[:, :T]
        x = x.reshape(B, T * P, self.d_model)

        act_emb = self.action_embed(actions).mean(dim=1)  # (B, d_model)

        causal = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, T, P, act_emb, causal_mask=causal)

        x = x.reshape(B, T, P, self.d_model)
        return self.head(self.ln_final(x))

model = MaskGITDynamics().to(device)
print(f"MaskGIT Dynamics params: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 5. Masked Training
# ============================================================
class MaskedTokenDataset(Dataset):
    def __init__(self, tokens, actions):
        """
        tokens: (N, T, P)
        actions: (N, T-1) or (N, T)
        """
        self.tokens = tokens
        self.actions = actions

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.actions[idx]


def mask_tokens(tokens, mask_ratio_range=(0.5, 1.0)):
    """
    tokens: (B, T, P) — mask only the LAST frame
    Returns: masked_tokens, mask (bool), target
    """
    B, T, P = tokens.shape
    masked = tokens.clone()
    target = tokens[:, -1, :].clone()  # (B, P) — last frame is the target

    # Random mask ratio per batch item
    mask_ratios = torch.rand(B) * (mask_ratio_range[1] - mask_ratio_range[0]) + mask_ratio_range[0]

    mask = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        n_mask = max(1, int(mask_ratios[b] * P))
        mask_indices = torch.randperm(P)[:n_mask]
        mask[b, mask_indices] = True
        masked[b, -1, mask_indices] = MASK_ID

    return masked, mask, target


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Assuming token_loader yields (tokens, actions) batches
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    count = 0

    for batch_tokens, batch_actions in token_loader:
        batch_tokens = batch_tokens.to(device)
        batch_actions = batch_actions.to(device)

        masked_tokens, mask, target = mask_tokens(batch_tokens)
        masked_tokens = masked_tokens.to(device)
        mask = mask.to(device)
        target = target.to(device)

        logits = model(masked_tokens, batch_actions)  # (B, T, P, vocab)
        last_logits = logits[:, -1]                    # (B, P, vocab)

        # Loss only on masked positions
        loss = F.cross_entropy(
            last_logits[mask],   # (num_masked, vocab)
            target[mask]          # (num_masked,)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/count:.4f}")

# ============================================================
# 6. MaskGIT Iterative Inference
# ============================================================
@torch.no_grad()
def generate_frame_maskgit(model, context_tokens, action_ids, num_steps=8):
    """
    context_tokens: (1, T_ctx, P) — known past frame tokens
    action_ids: (1, T_ctx + 1) — actions including for the new frame
    Returns: (1, P) predicted tokens for the new frame
    """
    P = context_tokens.shape[-1]

    # Start fully masked
    current = torch.full((1, 1, P), MASK_ID, device=device, dtype=torch.long)
    all_tokens = torch.cat([context_tokens, current], dim=1)  # (1, T_ctx+1, P)

    snapshots = []  # for visualization

    for step in range(num_steps):
        logits = model(all_tokens, action_ids)  # (1, T, P, vocab)
        next_logits = logits[:, -1]              # (1, P, vocab)

        probs = F.softmax(next_logits, dim=-1)
        max_probs, predicted = probs.max(dim=-1)  # (1, P)

        # How many to reveal this step (linear schedule — simple)
        total_to_reveal = int(P * (step + 1) / num_steps)
        already_revealed = (all_tokens[:, -1] != MASK_ID).sum().item()
        k = max(1, total_to_reveal - already_revealed)

        # Among still-masked positions, pick top-k by confidence
        is_masked = (all_tokens[:, -1] == MASK_ID)  # (1, P)
        confidence = max_probs.clone()
        confidence[~is_masked] = -1  # don't re-select unmasked

        n_masked = is_masked.sum().item()
        k = min(k, n_masked)

        if k > 0:
            _, topk_idx = confidence.topk(k, dim=-1)  # (1, k)
            for b in range(1):
                all_tokens[b, -1, topk_idx[b]] = predicted[b, topk_idx[b]]

        # Save snapshot for visualization
        snapshots.append(all_tokens[:, -1].clone())

    return all_tokens[:, -1], snapshots  # (1, P)

# ============================================================
# 7. Visualize Iterative Unmasking
# ============================================================
model.eval()

# Get seed context
# context_tokens: (1, 3, 64) — 3 seed frames tokenized
# action: (1, 4) — actions for all 4 frames

predicted_tokens, snapshots = generate_frame_maskgit(
    model, context_tokens, action_ids, num_steps=8
)

# Decode each snapshot to pixels
fig, axes = plt.subplots(1, len(snapshots) + 1, figsize=(3 * (len(snapshots) + 1), 3))

for i, snap in enumerate(snapshots):
    # Replace still-masked tokens with 0 for visualization
    vis_tokens = snap.clone()
    vis_tokens[vis_tokens == MASK_ID] = 0

    # Decode tokens → pixels using frozen tokenizer
    pixels = tokens_to_pixels(vis_tokens.unsqueeze(1))  # (1, 1, 3, 64, 64)
    img = pixels[0, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    n_revealed = (snap != MASK_ID).sum().item()
    axes[i].imshow(img)
    axes[i].set_title(f"Step {i+1}\n{n_revealed}/{N_PATCHES} revealed")
    axes[i].axis("off")

# Ground truth
gt_pixels = tokens_to_pixels(target_tokens.unsqueeze(1))
axes[-1].imshow(gt_pixels[0, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
axes[-1].set_title("Ground Truth")
axes[-1].axis("off")

plt.suptitle("MaskGIT: Iterative Frame Generation", fontsize=14)
plt.tight_layout()
plt.savefig("maskgit_steps.png", dpi=150)
plt.show()

# ============================================================
# 8. Speed Comparison
# ============================================================
print("\n--- Speed Comparison ---")

# MaskGIT: 8 steps
start = time.time()
for _ in range(10):
    generate_frame_maskgit(model, context_tokens, action_ids, num_steps=8)
maskgit_time = (time.time() - start) / 10
print(f"MaskGIT (8 steps): {maskgit_time*1000:.1f} ms per frame")

# Compare to autoregressive (would be ~64 sequential forward passes)
print(f"Autoregressive would be ~{64/8:.0f}x slower (64 steps vs 8)")
```

## What You Should Observe

1. **The crystallization effect** — early steps produce a blurry/blocky approximation, later steps fill in sharp details. Like going from a thumbnail to full resolution.

2. **~8x faster than autoregressive** — 8 forward passes instead of 64. Each pass processes ALL patches in parallel.

3. **Quality is comparable** — the iterative refinement compensates for the parallel prediction. High-confidence tokens are locked in early, providing context for harder predictions.

4. **Bidirectional attention matters** — unlike autoregressive, each patch can see other patches in the same frame. This gives more context for each prediction.

## Why This Matters

You now have all three components of TinyWorlds:
- **Video Tokenizer** (Iter 3-4): compress frames → discrete tokens
- **Latent Action Model** (Iter 7): infer actions without labels
- **MaskGIT Dynamics** (this): fast, controllable frame prediction

**Next iteration**: assemble everything into the complete pipeline.
