# Iteration 5: Answer

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# 2. Load Frozen Tokenizer (from Iteration 4)
# ============================================================
# --- Paste/import the SpaceTimeTokenizer, FSQ, SpaceTimeBlock classes from Iteration 4 ---
# (Omitted here for brevity — copy the classes from iter4 answer)
# Then:

# Option A: Load saved weights
# tokenizer = SpaceTimeTokenizer().to(device)
# tokenizer.load_state_dict(torch.load("../iter4_spacetime_transformer/tokenizer.pt"))

# Option B: Train fresh (if you haven't saved weights)
# ... train tokenizer as in Iteration 4, then continue below ...

tokenizer.eval()
for p in tokenizer.parameters():
    p.requires_grad = False  # Freeze

# ============================================================
# 3. Tokenize All Frames
# ============================================================
NUM_FRAMES = 4
FRAME_SKIP = 2
VOCAB_SIZE = 4 ** 5  # 1024
N_PATCHES = 64       # 8x8

print("Tokenizing all frames...")
all_tokens = []
with torch.no_grad():
    batch_size = 128
    for i in range(0, len(frames) - NUM_FRAMES * FRAME_SKIP, batch_size):
        clips = []
        for j in range(i, min(i + batch_size, len(frames) - NUM_FRAMES * FRAME_SKIP)):
            clip = frames[j:j + NUM_FRAMES * FRAME_SKIP:FRAME_SKIP]  # (T, 3, 64, 64)
            clips.append(clip)
        clips = torch.stack(clips).to(device)  # (B, T, 3, 64, 64)
        _, indices = tokenizer.encode(clips)    # (B, T, P)
        all_tokens.append(indices.cpu())

all_tokens = torch.cat(all_tokens, dim=0)  # (N, T, P)
print(f"Tokenized clips: {all_tokens.shape}")

# ============================================================
# 4. Token Sequence Dataset
# ============================================================
class TokenDataset(Dataset):
    def __init__(self, tokens):
        # tokens: (N, T, P) — each is a clip of T frames, P patches each
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        clip_tokens = self.tokens[idx]         # (T, P)
        context = clip_tokens[:-1]             # (T-1, P) — input
        target = clip_tokens[1:]               # (T-1, P) — shifted target
        return context, target

token_dataset = TokenDataset(all_tokens)
token_loader = DataLoader(token_dataset, batch_size=64, shuffle=True)

# ============================================================
# 5. Dynamics Model
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

        # FFN
        x = x + self.ffn(self.ln3(x))
        return x


class DynamicsModel(nn.Module):
    def __init__(self, vocab_size=1024, d_model=128, n_heads=4, n_layers=4,
                 n_patches=64, max_frames=4):
        super().__init__()
        self.d_model = d_model
        self.n_patches = n_patches

        # Token embedding (like word embedding in GPT)
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_patches, d_model) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # Classification head: predict which token from vocabulary
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_frames, max_frames), diagonal=1).bool()
        )

    def forward(self, tokens):
        """tokens: (B, T, P) integer indices → logits: (B, T, P, vocab_size)"""
        B, T, P = tokens.shape

        x = self.token_embed(tokens)  # (B, T, P, d_model)
        x = x + self.spatial_pos[:, :, :P] + self.temporal_pos[:, :T]
        x = x.reshape(B, T * P, self.d_model)

        causal = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, T, P, causal_mask=causal)

        x = x.reshape(B, T, P, self.d_model)
        logits = self.head(self.ln_final(x))  # (B, T, P, vocab_size)
        return logits

dynamics = DynamicsModel().to(device)
num_params = sum(p.numel() for p in dynamics.parameters())
print(f"Dynamics model parameters: {num_params:,}")

# ============================================================
# 6. Train
# ============================================================
optimizer = torch.optim.Adam(dynamics.parameters(), lr=1e-4)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for context, target in token_loader:
        context = context.to(device)  # (B, T-1, P)
        target = target.to(device)    # (B, T-1, P)

        logits = dynamics(context)    # (B, T-1, P, 1024)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/count:.4f}")

# ============================================================
# 7. Autoregressive Generation
# ============================================================
dynamics.eval()

def generate_frames(seed_tokens, num_generate=4):
    """seed_tokens: (1, T_seed, P) → generate num_generate new frames"""
    tokens = seed_tokens.clone()  # (1, T, P)

    for _ in range(num_generate):
        logits = dynamics(tokens)                    # (1, T, P, vocab)
        next_logits = logits[:, -1, :, :]            # (1, P, vocab) — last frame's prediction
        next_tokens = next_logits.argmax(dim=-1)     # (1, P) — greedy
        next_tokens = next_tokens.unsqueeze(1)       # (1, 1, P)
        tokens = torch.cat([tokens, next_tokens], dim=1)  # (1, T+1, P)

    return tokens  # (1, T_seed + num_generate, P)

def tokens_to_pixels(token_indices):
    """token_indices: (1, T, P) → (1, T, 3, 64, 64) using frozen tokenizer decoder"""
    # Convert indices back to quantized latents
    # indices → L-dimensional FSQ codes
    L = tokenizer.fsq.latent_dim
    basis = tokenizer.fsq.basis  # [1, 4, 16, 64, 256]

    # Decode indices to FSQ values
    z_q = torch.zeros(*token_indices.shape, L, device=token_indices.device)
    remaining = token_indices.clone().float()
    for i in range(L - 1, -1, -1):
        z_q[..., i] = torch.div(remaining, basis[i], rounding_mode='floor')
        remaining = remaining - z_q[..., i] * basis[i]

    return tokenizer.decode(z_q)

# Generate
with torch.no_grad():
    # Get a seed clip
    seed_clip = frames[:3 * FRAME_SKIP:FRAME_SKIP].unsqueeze(0).to(device)  # (1, 3, 3, 64, 64)
    seed_z_q, seed_tokens = tokenizer.encode(seed_clip)  # (1, 3, P)

    generated_tokens = generate_frames(seed_tokens, num_generate=4)  # (1, 7, P)
    generated_pixels = tokens_to_pixels(generated_tokens)             # (1, 7, 3, 64, 64)

# ============================================================
# 8. Visualize
# ============================================================
n_total = generated_pixels.shape[1]
fig, axes = plt.subplots(1, n_total, figsize=(3 * n_total, 3))

for t in range(n_total):
    img = generated_pixels[0, t].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    axes[t].imshow(img)
    label = f"Seed {t}" if t < 3 else f"Gen {t-3}"
    color = "green" if t < 3 else "red"
    axes[t].set_title(label, color=color)
    axes[t].axis("off")

plt.suptitle("Seed frames (green) → Generated frames (red)", fontsize=14)
plt.tight_layout()
plt.savefig("generated_frames.png", dpi=150)
plt.show()
```

## What You Should Observe

1. **Short-term predictions look reasonable** — the first 1-2 generated frames maintain the scene structure (ball, paddles, background).

2. **Drift over time** — after 3-4 generated frames, quality degrades. Colors shift, objects blur or disappear. Each prediction error compounds into the next.

3. **No control** — the model decides where the ball goes and how paddles move. You're just watching, not playing. There's no mechanism to input "move paddle up."

4. **This is GPT for video** — same architecture (transformer + causal attention + cross-entropy), same training (teacher forcing), same inference (autoregressive). Just a different vocabulary.

## Why This Matters

You've demonstrated that language modeling works for video. The frame prediction problem is solvable with the same tools as text prediction. But two problems remain:

1. **No controllability** — you need actions to make this interactive
2. **Slow decoding** — generating 64 tokens per frame one-by-one is sequential

**Next iteration**: add action conditioning using FiLM so you can control what happens in the generated world.
