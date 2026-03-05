# Iteration 9: Answer

This iteration is about integration. The answer is the assembled code from all previous iterations, organized into clean modules. Below is the key integration piece — the inference pipeline — since the individual model code comes from Iterations 2-8.

```python
# inference.py — Full TinyWorlds Inference Pipeline
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Import your trained models
from models.video_tokenizer import SpaceTimeTokenizer
from models.latent_actions import LatentActionModel, BinaryFSQ
from models.dynamics import MaskGITDynamics
from models.fsq import FSQ

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# 1. Load All Frozen Models
# ============================================================
# Stage 1: Video Tokenizer
tokenizer = SpaceTimeTokenizer(
    patch_size=8, d_model=128, n_heads=4, n_layers=2,
    latent_dim=5, num_bins=4, img_size=64, num_frames=4
).to(device)
tokenizer.load_state_dict(torch.load("checkpoints/tokenizer.pt", map_location=device))
tokenizer.eval()

# Stage 2: Latent Action Model
action_model = LatentActionModel(
    patch_size=8, d_model=128, n_heads=4,
    enc_layers=2, dec_layers=2, action_dim=3, img_size=64
).to(device)
action_model.load_state_dict(torch.load("checkpoints/action_model.pt", map_location=device))
action_model.eval()

# Stage 3: Dynamics
dynamics = MaskGITDynamics(
    vocab_size=1024, num_actions=8, d_model=128,
    n_heads=4, n_layers=4, n_patches=64, max_frames=4
).to(device)
dynamics.load_state_dict(torch.load("checkpoints/dynamics.pt", map_location=device))
dynamics.eval()

MASK_ID = 1024
N_PATCHES = 64
VOCAB_SIZE = 1024

# ============================================================
# 2. Helper Functions
# ============================================================
def tokens_to_pixels(token_indices):
    """Convert token indices → pixel frames using frozen tokenizer decoder."""
    L = tokenizer.fsq.latent_dim
    basis = tokenizer.fsq.basis

    z_q = torch.zeros(*token_indices.shape, L, device=device)
    remaining = token_indices.clone().float()
    for i in range(L - 1, -1, -1):
        z_q[..., i] = torch.div(remaining, basis[i], rounding_mode='floor')
        remaining = remaining - z_q[..., i] * basis[i]

    return tokenizer.decode(z_q)


@torch.no_grad()
def generate_frame_maskgit(context_tokens, action_id, num_steps=8):
    """Generate one new frame via MaskGIT iterative decoding."""
    B, T_ctx, P = context_tokens.shape

    # Start fully masked
    current = torch.full((B, 1, P), MASK_ID, device=device, dtype=torch.long)
    all_tokens = torch.cat([context_tokens, current], dim=1)
    T_total = all_tokens.shape[1]

    actions = torch.full((B, T_total), action_id, device=device, dtype=torch.long)

    for step in range(num_steps):
        logits = dynamics(all_tokens, actions)
        next_logits = logits[:, -1]  # (B, P, vocab)

        probs = F.softmax(next_logits, dim=-1)
        max_probs, predicted = probs.max(dim=-1)

        total_to_reveal = int(P * (step + 1) / num_steps)
        already_revealed = (all_tokens[:, -1] != MASK_ID).sum(dim=-1)

        for b in range(B):
            k = max(1, total_to_reveal - already_revealed[b].item())
            is_masked = (all_tokens[b, -1] == MASK_ID)
            confidence = max_probs[b].clone()
            confidence[~is_masked] = -1

            n_masked = is_masked.sum().item()
            k = min(k, n_masked)
            if k > 0:
                _, topk_idx = confidence.topk(k)
                all_tokens[b, -1, topk_idx] = predicted[b, topk_idx]

    return all_tokens[:, -1]  # (B, P)


# ============================================================
# 3. Load Seed Data
# ============================================================
import h5py

with h5py.File("data/pong_frames.h5", "r") as f:
    raw_frames = f["frames"][:]

frames_np = raw_frames.astype(np.float32) / 255.0
frames_all = np.transpose(frames_np, (0, 3, 1, 2))
frames_all = torch.from_numpy(frames_all)

# Pick a random starting point for seed frames
FRAME_SKIP = 2
NUM_SEED = 3
start = random.randint(0, len(frames_all) - NUM_SEED * FRAME_SKIP - 1)
seed_indices = list(range(start, start + NUM_SEED * FRAME_SKIP, FRAME_SKIP))
seed_frames = frames_all[seed_indices].unsqueeze(0).to(device)  # (1, 3, 3, 64, 64)

# Encode seed → tokens
with torch.no_grad():
    _, seed_tokens = tokenizer.encode(seed_frames)  # (1, 3, P)

# ============================================================
# 4. Generate Frames
# ============================================================
NUM_GENERATE = 12
context_tokens = seed_tokens.clone()

generated_pixels = []
actions_used = []

# Decode seed frames for display
with torch.no_grad():
    seed_pixels = tokens_to_pixels(seed_tokens)  # (1, 3, 3, 64, 64)
    for t in range(NUM_SEED):
        generated_pixels.append(seed_pixels[0, t].cpu())
        actions_used.append("seed")

print(f"Generating {NUM_GENERATE} frames...")
for step in range(NUM_GENERATE):
    # Choose action: cycle through some patterns
    if step < 4:
        action_id = 1  # up
    elif step < 8:
        action_id = 2  # down
    else:
        action_id = 0  # stay

    # Generate next frame
    next_tokens = generate_frame_maskgit(context_tokens, action_id, num_steps=8)

    # Decode to pixels
    with torch.no_grad():
        next_pixels = tokens_to_pixels(next_tokens.unsqueeze(1))
        generated_pixels.append(next_pixels[0, 0].cpu())
        actions_used.append(f"act={action_id}")

    # Slide context window
    context_tokens = torch.cat([
        context_tokens[:, 1:],
        next_tokens.unsqueeze(1)
    ], dim=1)

    if (step + 1) % 4 == 0:
        print(f"  Generated {step + 1}/{NUM_GENERATE} frames")

# ============================================================
# 5. Visualize — Full Generated Sequence
# ============================================================
n_total = len(generated_pixels)
cols = min(n_total, 8)
rows = (n_total + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
if rows == 1:
    axes = [axes]

for i, (pixels, label) in enumerate(zip(generated_pixels, actions_used)):
    row, col = i // cols, i % cols
    img = pixels.permute(1, 2, 0).clamp(0, 1).numpy()
    axes[row][col].imshow(img)
    color = "green" if label == "seed" else "blue"
    axes[row][col].set_title(label, fontsize=9, color=color)
    axes[row][col].axis("off")

# Hide empty subplots
for i in range(n_total, rows * cols):
    row, col = i // cols, i % cols
    axes[row][col].axis("off")

plt.suptitle("TinyWorlds: Seed (green) → Generated (blue)", fontsize=14)
plt.tight_layout()
plt.savefig("tinyworlds_generation.png", dpi=150)
plt.show()

# ============================================================
# 6. Compare Different Action Sequences
# ============================================================
print("\nGenerating with different action sequences...")
action_plans = {
    "All UP":    [1] * 8,
    "All DOWN":  [2] * 8,
    "All STAY":  [0] * 8,
    "Alternating": [1, 2, 1, 2, 1, 2, 1, 2],
}

fig, axes = plt.subplots(len(action_plans), 8 + NUM_SEED, figsize=(3 * (8 + NUM_SEED), 3 * len(action_plans)))

for row, (plan_name, action_list) in enumerate(action_plans.items()):
    ctx = seed_tokens.clone()

    # Show seed
    with torch.no_grad():
        seed_px = tokens_to_pixels(seed_tokens)
    for t in range(NUM_SEED):
        axes[row, t].imshow(seed_px[0, t].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[row, t].set_title("seed", fontsize=8, color="green")
        axes[row, t].axis("off")

    # Generate
    for step, act in enumerate(action_list):
        next_tok = generate_frame_maskgit(ctx, act, num_steps=8)
        with torch.no_grad():
            px = tokens_to_pixels(next_tok.unsqueeze(1))
        col = NUM_SEED + step
        axes[row, col].imshow(px[0, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
        axes[row, col].set_title(f"act={act}", fontsize=8, color="blue")
        axes[row, col].axis("off")

        ctx = torch.cat([ctx[:, 1:], next_tok.unsqueeze(1)], dim=1)

    axes[row, 0].set_ylabel(plan_name, fontsize=11, rotation=0, labelpad=70)

plt.suptitle("Same seed, different action sequences", fontsize=14)
plt.tight_layout()
plt.savefig("tinyworlds_action_comparison.png", dpi=150)
plt.show()

print("\nDone! Check tinyworlds_generation.png and tinyworlds_action_comparison.png")
```

## Training Scripts Summary

### train_stage1.py
```python
# Train SpaceTimeTokenizer on video clips
# Loss: smooth_l1_loss(reconstruction, original)
# Save: checkpoints/tokenizer.pt
```

### train_stage2.py
```python
# Train LatentActionModel on frame pairs
# Loss: l1_loss(predicted, actual) + 0.1 * variance_penalty
# Save: checkpoints/action_model.pt
```

### train_stage3.py
```python
# Load frozen tokenizer + action model
# Tokenize all training data → video tokens + action tokens
# Train MaskGITDynamics with masked prediction
# Loss: cross_entropy on masked positions
# Save: checkpoints/dynamics.pt
```

## What You Should Observe

1. **It works end-to-end** — raw pixels in, generated game world out.
2. **Different action sequences produce different futures** — the world responds to your inputs.
3. **Quality is reasonable for short horizons** — the first few generated frames are coherent. Longer sequences degrade.
4. **You understand every line** — because you built every component from scratch, motivated by problems you encountered firsthand.

## You're Done

You've rebuilt TinyWorlds. Every design choice in the original codebase now has a story:
- FSQ exists because continuous latents can't be predicted with cross-entropy (Iter 1→2)
- Space-time factored attention exists because full attention is too expensive (Iter 3→4)
- FiLM exists because actions need to modulate features (Iter 6)
- Latent actions exist because labels don't scale (Iter 6→7)
- MaskGIT exists because autoregressive is too slow (Iter 5→8)

Go explore the original `../tinyworlds/` codebase — it'll feel like reading code you wrote yourself.
